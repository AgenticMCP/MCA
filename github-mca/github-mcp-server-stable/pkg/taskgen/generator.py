"""Task-generation agent: realistic GitHub tasks grounded in the live catalog.

The generator does NOT invent tasks at random. It is forced through a
human-thinking scaffold (persona/motive -> concrete goal -> mental walkthrough
-> natural prompt) seeded with a scenario archetype, so the resulting prompt
reads like something a real user would ask, and its tool-call sequence is
recoverable by an independent planner. The generator's `expected_tools` is a
hypothesis used for coverage/diversity metadata only — ground truth is always
established downstream by execution + verification (see pipeline.py).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from pkg.agenticmcpe.config import LLMClient, extract_json

# Generation wants VARIETY, so it runs the LLM hotter than the planner/verifier
# (which need temperature 0 for determinism). At temperature 0 the generator
# collapses to near-identical tasks for the same archetype+difficulty; a higher
# temperature, combined with the sampled persona/phrasing below, is what makes
# successive tasks read like different real people. Used by the CLI when it
# builds the generator's LLM client.
GENERATION_TEMPERATURE = 0.85

# Personas and phrasing styles are sampled per task and injected into the prompt
# so generated tasks vary in voice, register and length instead of converging on
# one "assistant task" template. They steer wording only — never the tool choice.
PERSONAS: list[str] = [
    "an open-source maintainer triaging a busy backlog",
    "a developer evaluating a library before adopting it",
    "a release manager auditing what actually shipped",
    "a new contributor finding their way around an unfamiliar repo",
    "a tech lead preparing a short status update for their team",
    "a security-conscious engineer vetting a dependency",
    "a technical writer checking the docs against the code",
    "a student studying how a popular project is structured",
    "a site-reliability engineer wiring up CI for a fresh repo",
    "a project manager who just wants a quick, concrete answer",
]

PHRASING_STYLES: list[str] = [
    "terse and imperative — one direct sentence, no pleasantries",
    "conversational and polite — a couple of natural sentences, as if chatting",
    "context-first — a clause of background or motivation, then the concrete ask",
    "detailed and precise — names the exact repo, counts and paths up front",
    "slightly informal — natural wording, maybe a contraction or aside, still clear",
]

# Scenario archetypes: the human intents tasks are sampled from. `write` marks
# archetypes that require write tools (excluded in readonly mode).
ARCHETYPES: list[dict[str, Any]] = [
    {"name": "repo-exploration", "write": False,
     "intent": "Understand an unfamiliar public repository: find it, read key "
               "files (README, manifests), inspect branches, commits or contributors."},
    {"name": "issue-triage", "write": False,
     "intent": "Survey the health of a public repository's issue tracker: filter "
               "issues by state/label, read specific issues and their comments."},
    {"name": "pr-review", "write": False,
     "intent": "Review pull-request activity in a public repository: list PRs, "
               "inspect one PR's details, changed files, reviews or comments."},
    {"name": "code-search", "write": False,
     "intent": "Locate code: search code or repositories for a pattern/topic, "
               "then read one of the matching files."},
    {"name": "release-audit", "write": False,
     "intent": "Audit releases of a public repository: list tags or releases, "
               "read the latest release's notes, relate them to commit history."},
    {"name": "account-survey", "write": False,
     "intent": "Inspect the authenticated user's own account: profile details, "
               "owned repositories, notifications."},
    {"name": "repo-bootstrap", "write": True,
     "intent": "Start a new project under the authenticated user's account: "
               "create a repository, write initial files, create a feature "
               "branch, open a pull request."},
    {"name": "content-update", "write": True,
     "intent": "Create a repository with a file under the user's own account, "
               "then update that file's content (read its sha first), possibly "
               "via a branch and pull request."},
    {"name": "issue-reporting", "write": True,
     "intent": "Create a repository under the user's own account and file work "
               "items in it: create issues (optionally labeled), comment on "
               "them, close one."},
    {"name": "cross-repo-report", "write": True,
     "intent": "Gather data from well-known public repositories (issues, files, "
               "releases) and write a small report file into a new repository "
               "under the user's own account."},
    {"name": "automation-setup", "write": True,
     "intent": "Create a repository under the user's own account, add a GitHub "
               "Actions workflow file, and create the events (issue, push) that "
               "exercise it."},
    {"name": "pr-merge-flow", "write": True,
     "intent": "Ship a change the way a real contributor does, in a new "
               "repository under the user's own account: create the repository "
               "with an initial file, branch off it, commit the edit on the "
               "branch, open a pull request, then look the pull request up in "
               "the repository's open-PR list to confirm it is there before "
               "retitling it and merging it."},
    {"name": "stale-file-cleanup", "write": True,
     "intent": "Remove a file that is no longer needed from a new repository "
               "under the user's own account: create the repository, add a "
               "couple of files, list the directory to see what is there, read "
               "the obsolete file back to get its blob sha, then delete it with "
               "a clear commit message."},
    {"name": "backlog-grooming", "write": True,
     "intent": "Groom a small backlog in a new repository under the user's own "
               "account: create the repository, define a label, open a couple "
               "of issues using it, then list the open issues, comment on one "
               "with the outcome, and close it."},
    {"name": "upstream-mirror", "write": True,
     "intent": "Borrow a practice from a well-known public repository: read "
               "something real from it (its README, a workflow file, a manifest "
               "or its latest release notes), then create a repository under "
               "the user's own account and commit an adapted version of what "
               "was read, citing the upstream repo in the file or commit. The "
               "upstream repo is READ-ONLY here — never star, fork or otherwise "
               "write to it; every write lands in the user's own new repo."},
    {"name": "pr-code-review", "write": True,
     "intent": "Do a code review the way a reviewer actually does, on a pull "
               "request in a new repository under the user's own account: "
               "create the repository, put a change on a branch and open the "
               "pull request, then start a pending review on it, leave a line "
               "comment on the changed file, and submit the review as plain "
               "comments. The account opening the pull request is also the one "
               "reviewing it, and GitHub refuses to let an author approve or "
               "request changes on their own pull request — so the review is "
               "submitted as a COMMENT, never an approval."},
    {"name": "sub-issue-tracking", "write": True,
     "intent": "Break an epic into trackable pieces in a new repository under "
               "the user's own account: create the repository, open the parent "
               "issue, open the smaller follow-up issue, then attach the "
               "follow-up to the parent as a sub-issue so the parent shows its "
               "checklist."},
    {"name": "file-rename", "write": True,
     "intent": "Move or rename a file that is in the wrong place, in a new "
               "repository under the user's own account: create the repository "
               "with the file, read the file back to get its content, write "
               "that same content to the correct path, and delete the file at "
               "the old path so it is not left duplicated."},
    {"name": "ci-rerun", "write": True,
     "intent": "Take back a CI run that was started by mistake, in a new "
               "repository under the user's own account: create the repository "
               "with a workflow file, trigger the workflow, list the "
               "repository's workflow runs, and cancel the run that just "
               "started. Work only from the run listing — do not ask for a "
               "single run's details or its jobs."},
    {"name": "issue-correction", "write": True,
     "intent": "Fix a bug report that was filed in a hurry, in a new repository "
               "under the user's own account: create the repository, open the "
               "issue, then correct it afterwards — rewrite the title or body "
               "with the detail that was missing, and close it once the fix is "
               "described."},
    {"name": "label-refactor", "write": True,
     "intent": "Tidy a repository's label set, in a new repository under the "
               "user's own account: create the repository, add the two or three "
               "labels the project needs, then rename one of those labels it "
               "just created because the wording is off, and delete another one "
               "it just created that turned out to be redundant. Only ever "
               "rename or delete a label this same task created — a fresh repo "
               "has none of the labels you might assume."},
    {"name": "branch-sync", "write": True,
     "intent": "Bring a stale pull request up to date, in a new repository "
               "under the user's own account: create the repository, branch and "
               "open a pull request, land another commit on the default branch "
               "so the pull request falls behind, then look the pull request up "
               "in the open-PR list and update its branch from the base."},
    {"name": "multi-branch-bootstrap", "write": True,
     "intent": "Stand up a project repository the way a researcher or student "
               "kicks off real work, under the user's own account: create the "
               "repository, give it THREE to FIVE purposefully-named branches "
               "(a stable one plus per-person or per-workstream ones), put a "
               "README on the stable branch whose exact content the task "
               "quotes, seed a working branch by copying a real file (a "
               "setup.py, pyproject.toml or config) out of a well-known public "
               "repository, and open a pull request from that branch back to "
               "the stable one. Invent your own project domain, branch names "
               "and upstream source — do not reuse an example's."},
    {"name": "org-issue-report", "write": True,
     "intent": "Answer a question about an organisation's repositories and "
               "write the answer down, under the user's own account: search "
               "for that org's repositories matching a name pattern, and for "
               "each one count the issues in a particular state and label; "
               "then create a repository and commit the tally as a small JSON "
               "or CSV report whose filename and shape the task specifies. "
               "Pick your own org, name pattern and label."},
    {"name": "issue-bot-workflow", "write": True,
     "intent": "Build a little issue-answering robot, under the user's own "
               "account: create the repository with a README, add a GitHub "
               "Actions workflow that fires when an issue is opened and posts "
               "a DIFFERENT canned reply depending on the issue's label "
               "(including a fallback for unlabelled issues), then exercise it "
               "by filing two or three issues that hit the different branches "
               "of that rule. Choose your own labels and reply wording."},
    {"name": "comparative-fork", "write": True,
     "intent": "Pick a winner between well-known public repositories and adopt "
               "it: name two to four real ones, compare them on a concrete "
               "observable (fewest open issues, most recently created, most "
               "stars), fork whichever wins under the same name, then edit the "
               "fork's README to append a reference link to each of the "
               "repositories that lost. Read the losing repos only — the fork "
               "is the sole thing written to."},
    {"name": "audit-to-backlog", "write": True,
     "intent": "Turn an audit into actionable work, under the user's own "
               "account: search an organisation's repositories matching a name "
               "pattern, count each one's issues in a given state and label, "
               "then create a repository, commit the tally as a JSON or CSV "
               "report, define a triage label, and open one issue per finding "
               "so the numbers become a backlog someone can work."},
    {"name": "fork-and-report", "write": True,
     "intent": "Decide between real public repositories, adopt the winner, and "
               "write up the decision: compare two to four of them on a "
               "concrete observable (open issues, latest release date, stars), "
               "fork the winner under the same name, append a comparison table "
               "to the fork's README citing every candidate, and open an issue "
               "on the fork recording why it was chosen. The losing repos are "
               "read-only; the fork is the only thing written to."},
    {"name": "release-pipeline", "write": True,
     "intent": "Take a change from branch to shipped, under the user's own "
               "account: create the repository with a README and an Actions "
               "workflow on the default branch, cut a feature branch and commit "
               "on it, open a pull request, look it up in the open-PR list and "
               "merge it, then dispatch the workflow and read the run listing "
               "back to confirm it fired."},
    {"name": "collab-workspace", "write": True,
     "intent": "Set up a shared workspace for a small team, under the user's "
               "own account: create the repository with a README whose exact "
               "content the task quotes, cut a per-person branch for each "
               "teammate, seed each branch with a real file copied from a "
               "DIFFERENT well-known public repository, open a pull request "
               "from one teammate's branch, and label plus comment on it so "
               "the review has somewhere to start."},
    {"name": "two-repo-migration", "write": True,
     "intent": "Split work across TWO new repositories under the user's own "
               "account: create both, put files in the first, then copy the "
               "ones that belong elsewhere into the second by reading them back "
               "and writing them over, cross-link the two READMEs so each "
               "points at the other, and open an issue in each repo tracking "
               "what still has to move."},
    {"name": "stacked-prs", "write": True,
     "intent": "Land two dependent changes in order, under the user's own "
               "account: create the repository, cut TWO feature branches off "
               "the default one and commit a different file on each, open a "
               "pull request from each, then merge the first — which leaves the "
               "second behind — look the second up in the open-PR list, update "
               "its branch from the base, and merge it too."},
    {"name": "incident-response", "write": True,
     "intent": "Work a bug from report to closure, under the user's own "
               "account: create the repository with a file that contains the "
               "defect, file a labelled bug issue describing it, cut a hotfix "
               "branch, read the file back for its sha and commit the "
               "correction, open a pull request and merge it, then comment the "
               "resolution on the issue and close it."},
    {"name": "monorepo-restructure", "write": True,
     "intent": "Reorganise a repository's layout, under the user's own "
               "account: create it with several files sitting at the wrong "
               "paths, list the tree to see the mess, then move each one by "
               "reading its content, writing it at the correct path and "
               "deleting the original, and finish by committing a short note "
               "recording the new layout."},
    {"name": "issue-template-setup", "write": True,
     "intent": "Make a repository ready to receive contributions, under the "
               "user's own account: create it, then land the whole community "
               "scaffold in ONE multi-file commit — issue templates under "
               ".github, a CONTRIBUTING guide, a code of conduct — define the "
               "labels those templates refer to, and file one issue that "
               "follows a template to prove the setup works."},
    {"name": "bulk-triage", "write": True,
     "intent": "Clear a cluttered tracker, under the user's own account: "
               "create the repository, open several issues carrying different "
               "labels, then list them back and act on only the subset that "
               "matches one label — comment the disposition on each of those "
               "and close them, deliberately leaving the others open."},
    {"name": "watch-settings", "write": True,
     "intent": "Get on top of a noisy inbox, under the user's own account: "
               "create the repository with some content and an issue, adjust "
               "how you are subscribed to that repository's notifications, and "
               "clear the notification inbox so only future activity shows up."},
    {"name": "dependency-bump", "write": True,
     "intent": "Pin a dependency the way a maintainer does: read a real "
               "manifest (package.json, pyproject.toml, go.mod, Cargo.toml) "
               "out of a well-known public repository, create a repository "
               "under the user's own account holding an adapted manifest that "
               "pins an older version, then open an issue tracking the upgrade "
               "and a pull request on a branch that actually bumps the pin."},
    {"name": "docs-tree-scaffold", "write": True,
     "intent": "Lay out a documentation tree, under the user's own account: "
               "create the repository, commit several docs pages at once in a "
               "single multi-file commit, read the repository tree back to "
               "check the layout, then correct the one page filed at the wrong "
               "path by rewriting it where it belongs and deleting the stray."},
    {"name": "changelog-from-history", "write": True,
     "intent": "Write release notes from what actually landed, under the "
               "user's own account: create the repository and land a handful "
               "of separate commits with meaningful messages, then read that "
               "commit history back off the default branch and turn it into a "
               "CHANGELOG entry that reflects the real messages, proposing it "
               "on a branch via a pull request."},
    {"name": "parallel-branch-edits", "write": True,
     "intent": "Let one file diverge across branches and then compare, under "
               "the user's own account: create the repository with a config or "
               "settings file on the default branch, cut two or three branches "
               "and give the SAME path different content on each, then read "
               "that path back on every branch to see how they differ and open "
               "a pull request from whichever branch should win."},
    {"name": "security-policy-setup", "write": True,
     "intent": "Put a security policy in place, under the user's own account: "
               "create the repository, commit a SECURITY.md describing how to "
               "report a vulnerability plus a dependency-update config under "
               ".github, add a scheduled workflow that audits the project, and "
               "open a labelled issue tracking the first review."},
    {"name": "workflow-matrix-ci", "write": True,
     "intent": "Stand up more than one CI job and drive them, under the user's "
               "own account: create the repository with TWO manually-triggered "
               "workflow files that do different things, dispatch both, list "
               "the runs to see them queued, and cancel the one that was not "
               "needed — working only from the run listing."},
    {"name": "repo-handoff", "write": True,
     "intent": "Hand a project over to someone else, under the user's own "
               "account: create the repository, populate it with a few files "
               "and a couple of open issues, then read the repository tree back "
               "and commit a HANDOFF note that inventories what is there and "
               "what is outstanding, close the issues that are already done, "
               "and stop watching the repository."},
    {"name": "ci-inspection", "write": False,
     "intent": "Work out why CI is red on a public repository: list its recent "
               "workflow runs, pick a failing or latest run, inspect that run's "
               "jobs and read the log output for the job that matters."},
    {"name": "docs-fixup", "write": True,
     "intent": "Correct documentation already in flight in a new repository "
               "under the user's own account: create the repository with a docs "
               "file, spot the problem by reading the file back (which also "
               "yields its sha), commit the corrected content over it, and file "
               "an issue describing what was wrong."},
]

# Tools that depend on account capabilities/permissions the benchmark account
# lacks — steps using them always fail live, so tasks must never require them.
# Repo-level security/dependency alert tools need the security_events scope
# (and are never readable on third-party repos); Copilot's coding agent is not
# enabled on the account. Global security advisories remain fine (public data).
UNSUPPORTED_TOOLS = {
    "assign_copilot_to_issue",
    "request_copilot_review",
    "list_dependabot_alerts",
    "get_dependabot_alert",
    "list_code_scanning_alerts",
    "get_code_scanning_alert",
    "list_secret_scanning_alerts",
    "get_secret_scanning_alert",
    "list_repository_security_advisories",
    "list_org_repository_security_advisories",
}

# difficulty -> (min_steps, max_steps) for the mental walkthrough
DIFFICULTY_STEPS: dict[str, tuple[int, int]] = {
    "easy": (2, 3),
    "medium": (4, 6),
    "hard": (7, 10),
}

_GENERATOR_SYSTEM = """\
You are a TASK GENERATION agent. You invent realistic GitHub tasks that a human
would ask an AI assistant to perform. Each task is later given to an
INDEPENDENT planner that must rediscover the right tool-call sequence, execute
it against real GitHub, and verify the outcome; tasks that survive become
benchmark ground truth.

Think like a real GitHub user, in this exact order (the required human-thinking
structure):
1. PERSONA & MOTIVE — who is asking and why (maintainer triaging a backlog,
   developer evaluating a library, release manager auditing a release, ...).
2. CONCRETE GOAL — the specific outcome they want, with concrete names, numbers
   and limits (which repository, which label, how many items, what file name).
3. MENTAL WALKTHROUGH — how the human would do it by hand on github.com, as
   ordered steps ("first I look up X, that gives me Y, with Y I then ...").
   Each mental step must correspond to exactly ONE tool from the catalog, and
   each step's inputs must be obtainable from the task text or an earlier step.
4. TASK PROMPT — the request the persona would actually type, in natural
   language. It must be self-contained and unambiguous enough that the
   walkthrough can be reconstructed from the prompt alone.

Hard rules:
- expected_tools lists the catalog tool name of each mental step, in order.
  Use ONLY tool names from the provided catalog.
- The task prompt must NEVER contain tool names or the words "MCP"/"API"; it
  describes WHAT to achieve, not WHICH function to call.
- The task must be fully self-contained: it may not assume any pre-existing
  state on the authenticated user's account. If it updates or reports into
  something of the user's, it must create that thing first within the task.
- Read operations target well-known, long-lived public repositories (e.g.
  golang/go, python/cpython, microsoft/vscode, facebook/react, rust-lang/rust,
  torvalds/linux, github/github-mcp-server) so the data exists and is stable.
  Only ask about GitHub Releases in repositories that actually publish them
  (e.g. facebook/react, microsoft/vscode, cli/cli, github/github-mcp-server);
  golang/go and torvalds/linux use plain tags, not Releases.
- File paths are case-sensitive and must exist at the exact spelling. Only
  reference a specific file when you are certain of its path in that repo
  (README.md is safe in the example repos above; many repos spell it
  differently, e.g. chalk/chalk has "readme.md"). When unsure, formulate the
  task so the file is discovered first (list the directory, then read).
- Label names are repository-specific and unreliable to guess. The ONLY
  literal label you may name in a read task is 'bug' in microsoft/vscode. Any
  other label-related task MUST list the labels first and then drill into a
  label taken from that result (e.g. "the first label in the list").
- The executor can only PASS values between steps verbatim — it cannot
  compute, aggregate, reformat or combine data. Never ask for derived content
  (counts written into a report, summaries of fetched data, "a file listing
  the results"). File/issue content must be either literal text given in the
  task or the verbatim content of exactly one file read in an earlier step.
- GitHub never notifies a user of their OWN actions. Never design a task that
  expects a notification to appear because of something the task itself did,
  and never require dismissing/managing a notification (none may exist).
  Listing notifications as a pure read (possibly empty) is fine.
- Never involve GitHub Copilot (assigning Copilot to an issue, requesting a
  Copilot review): Copilot's coding agent is not enabled on this account, so
  those steps always fail.
- Never use repository-level security/dependency alert tools (Dependabot,
  code scanning, secret scanning alerts, repository security advisories):
  they require token permissions this account lacks and always 403 on
  third-party repositories. GLOBAL security advisories are public and fine.
- Keep result sizes bounded: ask for "up to N" items with N <= 10.
- The outcome must be verifiable afterwards by re-querying GitHub (counts,
  titles, file contents, states — not vague impressions).
- READONLY mode: every step uses a read-only tool; the task must not ask to
  create, modify or delete anything.
- WRITE mode: writes go ONLY to the authenticated user's own account (say
  "under your account"); any repository the task creates MUST have a name
  starting with "agenticmcpe-bench-". Never modify repositories the task did
  not itself create.
- Do not duplicate any of the EXISTING TASKS provided.

Naturalness (make it read like a real person, not a generated spec):
- Write the TASK PROMPT in the assigned persona's voice and phrasing style. Vary
  sentence length and register from task to task; never settle into one template.
- A real user says WHAT they want and maybe why — not an enumerated procedure.
  Never number the steps in the prompt, and never hint at how many calls or which
  tools it takes. The MENTAL WALKTHROUGH carries that structure, not the prompt.
- The prompt may include a short clause of natural context or motivation, but it
  must stay self-contained and unambiguous: the walkthrough must still be
  reconstructable from the prompt alone.

Output a SINGLE JSON object, no prose:
{
  "category": "<archetype name>",
  "thinking": ["<step 1 of the human walkthrough>", "..."],
  "task_prompt": "<the natural-language request>",
  "expected_tools": ["<tool>", "..."]
}
"""


class TaskGenError(RuntimeError):
    pass


@dataclass
class TaskSpec:
    category: str
    difficulty: str
    mode: str  # "readonly" | "write"
    thinking: list[str]
    task_prompt: str
    expected_tools: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "difficulty": self.difficulty,
            "mode": self.mode,
            "thinking": self.thinking,
            "task_prompt": self.task_prompt,
            "expected_tools": self.expected_tools,
            "warnings": self.warnings,
        }


class TaskGenerator:
    def __init__(self, llm: LLMClient, condensed_catalog: list[dict[str, Any]]):
        self.llm = llm
        self.catalog = condensed_catalog
        self._by_name = {c["tool"]: c for c in condensed_catalog}
        # token-cheap view for the generation prompt (no parameter schemas)
        self._gen_catalog = [
            {"tool": c["tool"], "ro": c["read_only"], "summary": c["summary"]}
            for c in condensed_catalog
        ]

    # ------------------------------------------------------------------ public
    def generate(
        self,
        archetype: dict[str, Any],
        *,
        difficulty: str = "medium",
        undercovered: list[str] | None = None,
        avoid: list[str] | None = None,
        attempts: int = 2,
    ) -> TaskSpec:
        """Generate one validated TaskSpec; retries once on a rejected spec."""
        mode = "write" if archetype["write"] else "readonly"
        # Sample the voice once per task (kept stable across the retry so a
        # rejection only fixes content, not flavour); varies across tasks.
        persona_pool = random.sample(PERSONAS, k=min(3, len(PERSONAS)))
        style = random.choice(PHRASING_STYLES)
        feedback = ""
        last: TaskGenError | None = None
        for _ in range(attempts):
            user = self._user_prompt(archetype, mode, difficulty,
                                     undercovered or [], avoid or [], feedback,
                                     persona_pool, style)
            reply = self.llm.chat(_GENERATOR_SYSTEM, user, json_mode=True)
            try:
                obj = extract_json(reply)
                return self._validate(obj, archetype, mode, difficulty, avoid or [])
            except (ValueError, TaskGenError) as e:
                last = e if isinstance(e, TaskGenError) else TaskGenError(str(e))
                feedback = (f"\n\nYour previous attempt was REJECTED for this "
                            f"reason:\n{last}\nGenerate a corrected task.")
        assert last is not None
        raise last

    # ---------------------------------------------------------------- internal
    def _user_prompt(self, archetype: dict[str, Any], mode: str, difficulty: str,
                     undercovered: list[str], avoid: list[str], feedback: str,
                     persona_pool: list[str], style: str) -> str:
        lo, hi = DIFFICULTY_STEPS[difficulty]
        parts = [
            "TOOL CATALOG (authoritative; 'ro' = read-only):",
            json.dumps(self._gen_catalog, separators=(",", ":")),
            "",
            f"SCENARIO ARCHETYPE: {archetype['name']} — {archetype['intent']}",
            f"MODE: {mode}",
            f"DIFFICULTY: {difficulty} — the walkthrough should need {lo}-{hi} tool calls.",
            f"PERSONA — adopt whichever best fits this archetype: {persona_pool}",
            f"PHRASING STYLE — shape the prompt's register and length like this: {style}",
        ]
        if undercovered:
            parts.append(
                "UNDER-COVERED TOOLS (prefer exercising 1-2 of these when it fits "
                f"the archetype naturally; never force an unnatural fit): {undercovered[:15]}"
            )
        if avoid:
            parts.append("EXISTING TASKS (do not duplicate):")
            parts.append(json.dumps(avoid[-20:], ensure_ascii=False))
        parts.append("\nGenerate ONE task now.")
        return "\n".join(parts) + feedback

    def _validate(self, obj: Any, archetype: dict[str, Any], mode: str,
                  difficulty: str, avoid: list[str]) -> TaskSpec:
        if not isinstance(obj, dict):
            raise TaskGenError(f"generator output is not an object: {obj!r}")
        prompt = str(obj.get("task_prompt") or "").strip()
        tools = obj.get("expected_tools") or []
        if len(prompt) < 40:
            raise TaskGenError("task_prompt is missing or too short")
        if not isinstance(tools, list) or not tools:
            raise TaskGenError("expected_tools is missing or empty")
        unknown = [t for t in tools if t not in self._by_name]
        if unknown:
            raise TaskGenError(f"expected_tools not in catalog: {unknown}")
        unsupported = [t for t in tools if t in UNSUPPORTED_TOOLS]
        if unsupported:
            raise TaskGenError(
                f"expected_tools use account-unsupported tools: {unsupported}")
        if mode == "readonly":
            writers = [t for t in tools if not self._by_name[t]["read_only"]]
            if writers:
                raise TaskGenError(f"readonly mode but expected write tools: {writers}")
        leaked = [t for t in self._by_name if t in prompt and "_" in t]
        if leaked:
            raise TaskGenError(f"task_prompt leaks literal tool names: {leaked}")
        norm = prompt.casefold()
        if any(norm == a.casefold() for a in avoid):
            raise TaskGenError("task_prompt duplicates an existing task")

        warnings: list[str] = []
        lo, hi = DIFFICULTY_STEPS[difficulty]
        if not lo <= len(tools) <= hi:
            warnings.append(
                f"expected {lo}-{hi} steps for {difficulty}, got {len(tools)}")
        return TaskSpec(
            category=str(obj.get("category") or archetype["name"]),
            difficulty=difficulty,
            mode=mode,
            thinking=[str(t) for t in (obj.get("thinking") or [])],
            task_prompt=prompt,
            expected_tools=[str(t) for t in tools],
            warnings=warnings,
        )


__all__ = ["ARCHETYPES", "DIFFICULTY_STEPS", "TaskGenError", "TaskGenerator", "TaskSpec"]
