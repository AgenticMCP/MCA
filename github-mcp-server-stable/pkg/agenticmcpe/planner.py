"""Planner agent: task prompt -> grounded tool-call sequence (s0..sn) + summary.

The planner is *grounded*: it only ever sees the real github-mcp-server tool
catalog (names, one-line docs, parameter schemas), so it cannot invent tools or
parameters. After the LLM proposes a plan we validate every step against the
catalog (tool exists, required params present) before writing it out — cheap
local rejection beats a failed server round-trip.

Output artifacts (written under ``settings.work_dir``):

* ``plan.json``  — machine-readable sequence the executor replays verbatim.
* ``plan.md``    — human summary of the sequence with inputs/expected outputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import LLMClient, Settings, extract_json
from .rag import (KBRetriever, RetrievedExample, catalog_enum_index,
                  task_grounds_entry)

# A binding is a whole-value string "$<step_id>" optionally followed by a path.
_BINDING_REF_RE = re.compile(r"^\$(?P<step>[A-Za-z_][A-Za-z0-9_-]*)(?:[.\[].*)?$")

# RAG (knowledge-base) retrieval thresholds, all tunable:
# - TOP_K        how many solved examples to retrieve.
# - STRONG       cosine >= this AND every literal grounded in the task => REUSE
#                the stored sequence directly, with no LLM planning call.
# - EXAMPLE_FLOOR below this a hit is noise and is not even shown to the LLM.
RAG_TOP_K = 3
RAG_STRONG_THRESHOLD = 0.86
RAG_EXAMPLE_FLOOR = 0.18

# JSON-Schema scalar type -> accepted python types, for local plan validation.
_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _binding_refs(value: Any) -> set[str]:
    """Collect the step ids referenced by binding templates inside ``value``."""
    refs: set[str] = set()
    if isinstance(value, str):
        m = _BINDING_REF_RE.match(value)
        if m:
            refs.add(m.group("step"))
    elif isinstance(value, dict):
        for v in value.values():
            refs |= _binding_refs(v)
    elif isinstance(value, list):
        for v in value:
            refs |= _binding_refs(v)
    return refs

_PLANNER_SYSTEM = """\
You are the PLANNER in an agentic workflow that replaces a human driving the \
GitHub MCP server. You translate a natural-language task into a STRICT, ordered \
tool-call sequence using ONLY the provided tool catalog.

Rules:
- Use ONLY tools that appear in the catalog. Never invent tools or parameters.
- Only use parameter names that appear in a tool's `params`. Provide every \
required parameter.
- Steps are ordered and identified s0, s1, ... A later step may reference an \
earlier step's output with a binding string "$<id>.<json.path>" \
(e.g. "$s0.number", "$s1.items[0].id"). A binding that is the WHOLE value of a \
parameter keeps its native type (numbers stay numbers). A binding may ALSO be \
EMBEDDED inside a longer string: every "$sN.path" segment is substituted with \
the value rendered as text at execution time — e.g. a search query \
"repo:$s1.items[0].full_name is:issue" or report content \
"$s1.items[0].full_name,$s2.totalCount\\n". Build queries, report lines and \
file content from prior outputs this way; NEVER hand-copy or guess a value \
that a prior step returns.
- Prefer the minimum number of steps that accomplishes the task. Do not add \
speculative or cleanup steps unless the task requires them.
- For each step include `post_action_properties`: a JSON object of checkable \
expectations about the result of that step, that a verifier can assert later. \
Use these keys when relevant:
    - "expect_fields": [list of field names the result object must contain]
    - "invariants": { "field": expected_fixed_value, ... }  (values that must equal the input/known constant)
    - "bounds": { "field": {"min": n, "max": m} }  (numeric boundaries)
    - "metamorphic": "short natural-language relation to re-check live, e.g. 'an issue with this number now exists and is open'"

GitHub write-task guardrails (follow exactly):
- The repository owner is the authenticated user. ALWAYS make `get_me` the first
  step (s0) and reference the owner as the binding "$s0.login". NEVER hardcode an
  owner or use a placeholder like "theuser"/"owner"/"your-username".
- NEVER invent values (owners, repos, SHAs, branch names) you were not given or
  did not obtain from a prior step. Anything produced by a prior step MUST be a
  binding.
- External/third-party repositories: when the task names an exact owner AND
  repo, use them directly — do NOT search for them. `search_repositories` does
  NOT accept the "repo:owner/name" qualifier (the API rejects it with 422); to
  discover repos, scope the query like "org:google generative-ai in:name". If a
  repo the task names turns out not to exist (404 / "Could not resolve"), the
  task text may be misspelled — find the real repo with `search_repositories`
  by name and bind owner/name from the result.
- Any step that uses a repo FOUND BY A SEARCH step must bind BOTH
  "$sN.items[i].owner.login" AND "$sN.items[i].name". Never pair a searched
  repo name with an owner you assumed from the task text — repositories get
  renamed and transferred across orgs, so the real owner is whatever the
  search result says.
- To COUNT or LIST a specific repository's issues, call `list_issues` with
  state/labels (add perPage:1 when only the count matters) and bind
  "$sN.totalCount". Do NOT use `search_issues` with a "repo:" qualifier — it
  fails with 422 for renamed/old repo names, while list_issues follows
  renames. Use search_issues only for cross-repo text searches.
- `fork_repository` forks a RENAMED source under its CURRENT name, and its
  result carries the fork's real location — in every later step bind the
  fork's owner/name from the fork step's result ("$sN.owner.login",
  "$sN.name"); never assume the fork kept the name you requested.
- NEVER bind a search item beyond items[0]. How many items a search returns
  is unknown when you plan — "$sN.items[2]" of a search that finds 2 repos is
  a fatal binding error, and result ORDER is not the task's listing order
  anyway. When the task involves SEVERAL repositories, plan one tightly-scoped
  search per repository (org:/user: + in:name) and bind items[0] of each;
  never fan multiple entities out of a single search's item list.
- A search may match NOTHING (zero items). If execution reports an empty
  items list, broaden the query on replan (drop qualifiers, fewer words) —
  never keep a plan that binds items[0] of a search that returned nothing.
- Every step must be one you EXPECT TO SUCCEED. Execution is strict and
  in-order: the first failing step aborts the plan and every later step is
  skipped — there are no fallbacks, so a "try this owner, then search" probe
  kills the run at the probe. When the owner or exact repo name is uncertain
  (the task names it loosely or possibly misspelled), RESOLVE FIRST with one
  scoped search, bind "$sN.items[0].owner.login"/"$sN.items[0].name", and only
  then fetch or write — never place a guessed-owner step ahead of the search
  that would have resolved it.
- Bindings are LOOKUPS ONLY — there are NO expressions, comparisons, ternaries
  or arithmetic ("$s1.total_count <= $s2.total_count ? 'A' : 'B'" is invalid
  and fails). When an action depends on COMPARING runtime values, include the
  read steps plus the action with your best-guess literal; if the guess is
  wrong it fails, and on replan the per-step outputs show both values — then
  hardcode the correct literal.
- To create a repository whose README/initial files you control, call
  `create_repository` with `autoInit:false`, then create README.md with
  `create_or_update_file` (this also creates the default branch "main"). Do NOT
  combine `autoInit:true` with creating README.md — autoInit pre-creates it and a
  second create without its `sha` fails with "already exists".
- `create_branch` requires the repo and its base branch to already exist. Create
  the repo and write at least one file to "main" before branching from it.
- When the task supplies literal file content (quoted text, often introduced
  with words like "with the content" / "exact content"), reproduce it
  VERBATIM in the write step — preserve its whitespace and apparent typos, do
  NOT "fix" or reformat it. The verifier compares the live file byte-for-byte
  against the task's quoted text.
- To UPDATE a file that already exists — including a file INHERITED by a
  branch that was created from main — you MUST first `get_file_contents` on
  that exact path with `ref` set to the target branch, then pass "$sN.sha" to
  `create_or_update_file`. Writing to an existing path without `sha` always
  fails with "File already exists".
- `get_file_contents` returns an OBJECT {content, sha}: `content` is the file's
  text — bind "$sN.content" (e.g. to copy a file into another repo); `sha` is
  its blob SHA — bind "$sN.sha" when UPDATING that file with
  `create_or_update_file`. When the path is ambiguous/not found, `content`
  instead holds a disambiguation message listing the real path (e.g.
  "matching files: ['chat/train.py']"). If you are unsure of a file's exact
  path in another repo, expect the first lookup to reveal it, then fetch that
  exact path.
- Result shapes for list bindings (authoritative — never guess others):
  `list_issues` returns {issues:[...], totalCount, pageInfo} -> bind
  "$sN.issues[0].number"; `list_label` returns {labels:[...], totalCount} ->
  bind "$sN.labels[0].name"; the search_* tools return {total_count,
  items:[...]} -> bind "$sN.items[0]..."; `list_commits`, `list_branches`,
  `list_pull_requests`, `list_releases`, `list_tags` and `list_notifications`
  return a bare ARRAY -> bind "$sN[0]..."; `actions_list` returns
  {workflow_runs:[...]} and a run's identifier field is `id` -> bind
  "$sN.workflow_runs[0].id"; `get_tag` returns {tag, sha, object:{sha, type},
  ...} where the COMMIT the tag points to is "$sN.object.sha" (".sha" alone is
  the tag object itself, and there is no ".commit").
- `issue_write` (method create) returns {id, url, number} -> bind
  "$sN.number" as the issue_number for follow-up steps (comments, updates,
  closing). The `id` field is NOT the issue number — never use it as one.
- Notifications: a user is NEVER notified of their own actions, so
  `list_notifications` is usually EMPTY right after this plan's own writes —
  never bind into its elements expecting one to exist.
- A binding "$sN..." may only reference a step that appears EARLIER in this same
  plan. Never bind to a step id you did not include.
- Order matters: create repo -> write main files -> create branches -> write
  branch files -> open pull request.
- When you generate a GitHub Actions workflow (a file under `.github/workflows/`)
  that comments, labels, or otherwise WRITES via the built-in GITHUB_TOKEN, you
  MUST add a top-level `permissions:` block granting what it needs (e.g.
  `permissions:\\n  issues: write`), because a new repo's default token is
  read-only and writes will 403. In `actions/github-script`, `await` the async
  API calls.
- ANY file under `.github/workflows/` MUST be written with branch set to the
  repository's DEFAULT branch ("main"). GitHub triggers issue/label/comment
  workflows ONLY from the default branch — a workflow pushed to a dev/feature
  branch NEVER runs, even when the task stages its other files on that branch.
- Output a SINGLE JSON object, no prose, with this exact shape:
{
  "summary": "<one paragraph describing what the sequence does>",
  "steps": [
    {
      "id": "s0",
      "tool": "<tool name from catalog>",
      "arguments": { ... },
      "post_action_properties": { ... },
      "description": "<why this step>"
    }
  ]
}
"""

_REPLAN_NOTE = """\

The previous plan FAILED during execution. Below is the prior plan with the \
per-step outcome and, for read steps, what they returned.

Produce a FRESH plan (renumber from s0) that completes the ORIGINAL task FROM \
THE CURRENT STATE. Requirements:
- SUCCESS steps already happened; their repositories, branches and files ALREADY \
EXIST. Do NOT include steps that re-create them.
- You MAY (and usually must) RE-INCLUDE read steps such as get_me and \
get_file_contents, because later steps bind to their outputs. Anything you bind \
to with "$sN" MUST be a step you include in this plan — e.g. if you set an owner \
from the login, include get_me; if you write a file's content from another file, \
include the get_file_contents that produced it.
- Fix what failed using the read-step outputs below — e.g. if get_file_contents \
returned a disambiguation message naming the real path (like "chat/train.py"), \
call get_file_contents again with that EXACT path before writing the file.
- The step outputs below are TRUNCATED PREVIEWS, never complete file contents. \
NEVER paste content from this report into a write step as a literal — a file \
whose content came from fetching another file must ALWAYS be written from a \
fresh `get_file_contents` step's binding ("$sN.content") in THIS plan. \
Retyping, summarising or abbreviating fetched content writes a corrupt file \
that verification fails byte-for-byte.
- Resources may ALSO survive from an EARLIER run of this same task, not only \
from the steps above. An "already exists" style error means the resource is \
ALREADY THERE: reference it (bind the number/name the report shows, or add a \
read/list step to look it up) — NEVER add a step that creates it again. In \
particular never re-create an issue, comment, gist or pull request the report \
says exists; that would duplicate it.
- NEVER resubmit the failed plan unchanged — materially change the failing \
step (different arguments, a corrected binding, or an extra lookup step) based \
on the error text. If the error shows a literal "$sN..." string reaching \
GitHub, that binding's path was invalid: bind ONLY fields the step outputs \
below actually show.

EXECUTION REPORT:
{feedback}
"""


@dataclass
class PlanStep:
    id: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    post_action_properties: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    expect_output: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, obj: dict[str, Any], default_id: str) -> "PlanStep":
        raw_args = dict(obj.get("arguments") or obj.get("args") or {})
        return cls(
            id=str(obj.get("id") or default_id),
            tool=obj["tool"],
            # Models emit `null` for optional params they want to skip; the
            # schema rejects nulls, so omitting is the only correct encoding.
            arguments={k: v for k, v in raw_args.items() if v is not None},
            post_action_properties=dict(obj.get("post_action_properties") or {}),
            description=obj.get("description", ""),
            expect_output=obj.get("expect_output"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "arguments": self.arguments,
            "post_action_properties": self.post_action_properties,
            "description": self.description,
            **({"expect_output": self.expect_output} if self.expect_output else {}),
        }


@dataclass
class Plan:
    task: str
    summary: str
    steps: list[PlanStep]
    provider: str = ""
    model: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "provider": self.provider,
            "model": self.model,
            "summary": self.summary,
            "warnings": self.warnings,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Plan":
        steps = [PlanStep.from_dict(s, f"s{i}") for i, s in enumerate(obj.get("steps", []))]
        return cls(
            task=obj.get("task", ""),
            summary=obj.get("summary", ""),
            steps=steps,
            provider=obj.get("provider", ""),
            model=obj.get("model", ""),
            warnings=list(obj.get("warnings", [])),
        )


class PlannerError(RuntimeError):
    pass


class PlannerAgent:
    def __init__(self, llm: LLMClient, condensed_catalog: list[dict[str, Any]],
                 retriever: "KBRetriever | None" = None):
        self.llm = llm
        self.catalog = condensed_catalog
        self._by_name = {c["tool"]: c for c in condensed_catalog}
        # Enum/method values from the schema, exempted by the reuse gate below.
        self._enum_index = catalog_enum_index(condensed_catalog)
        # Optional RAG corpus of verified task->sequence mappings. When None the
        # planner behaves exactly as before (pure LLM planning). The taskgen
        # pipeline deliberately leaves this None so it keeps rediscovering
        # sequences independently — the basis of its ground truth.
        self.retriever = retriever

    # ------------------------------------------------------------------ public
    def plan(self, task: str, *, feedback: str | None = None) -> Plan:
        # --- RAG: consult the knowledge base BEFORE the LLM. Only on the initial
        # plan (a state-aware replan must reason about live execution state, not
        # a precedent), and only when a retriever is wired and non-empty. ---
        examples: list[RetrievedExample] = []
        if self.retriever is not None and not feedback and not self.retriever.is_empty:
            examples = self.retriever.retrieve(task, k=RAG_TOP_K)
            reused = self._maybe_reuse(task, examples)
            if reused is not None:
                return reused  # sequence "referable" -> no LLM call needed

        system = _PLANNER_SYSTEM
        catalog_json = json.dumps(self.catalog, separators=(",", ":"))
        user = f"TOOL CATALOG (authoritative):\n{catalog_json}\n\nTASK:\n{task}\n"
        user += self._format_examples(examples)
        if feedback:
            user += _REPLAN_NOTE.format(feedback=feedback)

        reply = self.llm.chat(system, user, json_mode=True)
        try:
            obj = extract_json(reply)
        except ValueError as e:
            raise PlannerError(f"planner produced unparseable output: {e}") from e
        if not isinstance(obj, dict) or "steps" not in obj:
            raise PlannerError(f"planner output missing 'steps': {obj!r}")

        steps = [
            PlanStep.from_dict(s, f"s{i}") for i, s in enumerate(obj.get("steps", []))
        ]
        plan = Plan(
            task=task,
            summary=str(obj.get("summary", "")),
            steps=steps,
            provider=self.llm.settings.provider,
            model=self.llm.settings.model,
        )
        plan.warnings = self._validate(plan)
        return plan

    # ----------------------------------------------------------------- RAG path
    def _maybe_reuse(self, task: str, hits: list[RetrievedExample]) -> Plan | None:
        """If the top KB hit is a near-identical task whose literal arguments are
        all grounded in THIS task, reuse its verified sequence verbatim — that is
        what "the tool-call sequence can be referred" means. Returns the reused
        Plan, or None to fall through to LLM planning. The reused plan is still
        re-validated against the live catalog, so a stale entry (a tool/param
        that no longer exists) safely falls back instead of being trusted."""
        if not hits:
            return None
        top = hits[0]
        if top.score < RAG_STRONG_THRESHOLD or not task_grounds_entry(
                top.entry, task, self._enum_index):
            return None
        steps_raw = top.entry.get("steps") or []
        if not steps_raw:
            return None
        steps = [PlanStep.from_dict(s, f"s{i}") for i, s in enumerate(steps_raw)]
        plan = Plan(
            task=task,
            summary=str(top.entry.get("task_summary", "")),
            steps=steps,
            provider=self.llm.settings.provider,
            model=self.llm.settings.model,
        )
        try:
            plan.warnings = self._validate(plan)
        except PlannerError:
            return None  # stored sequence no longer valid -> let the LLM replan
        plan.warnings.insert(
            0, f"RAG: reused verified KB entry {top.entry.get('id')} "
               f"(similarity={top.score:.2f}); no LLM planning call was made")
        return plan

    def _format_examples(self, hits: list[RetrievedExample]) -> str:
        """Render retrieved solved tasks as in-context precedent for the LLM.
        Empty string when there is nothing worth showing."""
        useful = [h for h in hits if h.score >= RAG_EXAMPLE_FLOOR]
        if not useful:
            return ""
        items = [
            {
                "similar_task": h.entry.get("task_prompt", ""),
                "tool_sequence": h.entry.get("tool_sequence", []),
                "steps": _compact_steps(h.entry.get("steps") or []),
            }
            for h in useful
        ]
        return (
            "\n\nSOLVED EXAMPLES retrieved from the verified knowledge base (most "
            "similar first). Treat them as PRECEDENT, not the answer: follow the "
            "tool choice and ORDER when the task shape matches, but RE-DERIVE every "
            "argument from THIS task and from your own earlier steps via $bindings "
            "— never copy another task's literal owners, repos, paths or SHAs:\n"
            + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        )

    def write(self, plan: Plan, settings: Settings) -> tuple[str, str]:
        """Persist plan.json + plan.md. Returns their paths."""
        settings.ensure_work_dir()
        json_path = settings.work_dir / "plan.json"
        md_path = settings.work_dir / "plan.md"
        json_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        md_path.write_text(self._render_md(plan), encoding="utf-8")
        return str(json_path), str(md_path)

    # ------------------------------------------------------------------ internal
    def _validate(self, plan: Plan) -> list[str]:
        """Reject hallucinated tools/params and dangling bindings before
        execution. Hard errors raise; soft issues are returned as warnings."""
        warnings: list[str] = []
        seen_ids: set[str] = set()
        for st in plan.steps:
            if st.id in seen_ids:
                raise PlannerError(f"duplicate step id {st.id!r}")
            # binding references must point to an EARLIER step in this plan
            for ref in _binding_refs(st.arguments):
                if ref not in seen_ids:
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): binding '${ref}...' references "
                        f"step {ref!r}, which is not an earlier step in this plan "
                        f"(earlier ids: {sorted(seen_ids)})"
                    )
            seen_ids.add(st.id)
            spec = self._by_name.get(st.tool)
            if spec is None:
                raise PlannerError(
                    f"step {st.id}: tool {st.tool!r} is not in the catalog"
                )
            known = {p["name"] for p in spec["params"]}
            required = {p["name"] for p in spec["params"] if p["required"]}
            present = set(st.arguments)
            missing = required - present
            if missing:
                raise PlannerError(
                    f"step {st.id} ({st.tool}): missing required params {sorted(missing)}"
                )
            unknown = present - known
            if unknown:
                warnings.append(
                    f"step {st.id} ({st.tool}): params not in schema {sorted(unknown)} "
                    "(server may reject)"
                )
            # enum/type check literal scalar args (bindings resolve later) —
            # catching these locally turns an execution-time validation
            # failure into a free planner self-correction.
            by_name = {p["name"]: p for p in spec["params"]}
            for k, v in st.arguments.items():
                p = by_name.get(k)
                if p is None or (isinstance(v, str) and _BINDING_REF_RE.match(v)):
                    continue
                if p.get("enum") and isinstance(v, (str, int)) and v not in p["enum"]:
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): param {k}={v!r} not in enum "
                        f"{p['enum']}")
                expected = _TYPE_MAP.get(p.get("type"))
                if expected and not isinstance(v, expected):
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): param {k}={v!r} must be of "
                        f"type {p['type']!r}")
                if p.get("type") == "integer" and isinstance(v, bool):
                    raise PlannerError(
                        f"step {st.id} ({st.tool}): param {k}={v!r} must be an integer")
        return warnings

    @staticmethod
    def _render_md(plan: Plan) -> str:
        lines = [
            f"# Plan for task\n",
            f"> {plan.task}\n",
            f"**Provider/model:** {plan.provider} / {plan.model}\n",
            f"## Summary\n\n{plan.summary}\n",
        ]
        if plan.warnings:
            lines.append("## Warnings\n")
            lines += [f"- {w}" for w in plan.warnings]
            lines.append("")
        lines.append("## Tool-call sequence\n")
        for st in plan.steps:
            lines.append(f"### {st.id} — `{st.tool}`")
            if st.description:
                lines.append(f"{st.description}\n")
            lines.append("```json")
            lines.append(json.dumps(st.arguments, indent=2))
            lines.append("```")
            if st.post_action_properties:
                lines.append("**Post-action properties:**")
                lines.append("```json")
                lines.append(json.dumps(st.post_action_properties, indent=2))
                lines.append("```")
            lines.append("")
        return "\n".join(lines)


def _compact_steps(steps: list[dict[str, Any]], max_val: int = 120) -> list[dict[str, Any]]:
    """Trim a KB entry's steps for the prompt: keep tool + arguments, truncating
    long string values (e.g. embedded file content) so examples stay token-cheap."""
    out: list[dict[str, Any]] = []
    for s in steps:
        args: dict[str, Any] = {}
        for k, v in (s.get("arguments") or {}).items():
            if isinstance(v, str) and len(v) > max_val:
                args[k] = v[:max_val] + "…<truncated>"
            else:
                args[k] = v
        out.append({"tool": s.get("tool"), "arguments": args})
    return out


__all__ = ["PlanStep", "Plan", "PlannerAgent", "PlannerError"]
