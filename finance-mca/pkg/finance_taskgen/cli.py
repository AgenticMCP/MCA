"""CLI for the finance taskgen benchmark builder.

    python -m pkg.finance_taskgen generate --count 3              # full loop
    python -m pkg.finance_taskgen generate --count 2 --dry-run    # prompts only
    python -m pkg.finance_taskgen stats                           # KB coverage
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from typing import Sequence

from pkg.finance_agenticmcpe.config import AgenticConfig, default_config
from pkg.finance_agenticmcpe.llm import LLMClient, LLMError
from pkg.finance_agenticmcpe.planner import _condensed_catalog
from pkg.finance_agenticmcpe.catalog import discover

from .generator import (
    ARCHETYPES,
    DIFFICULTY_STEPS,
    GENERATION_TEMPERATURE,
    TaskGenError,
    TaskGenerator,
)
from .kb import KnowledgeBase
from .pipeline import BenchmarkPipeline

_DIFF_CYCLE = ["medium", "hard", "expert", "hard"]


def cmd_generate(args: argparse.Namespace) -> int:
    kb = KnowledgeBase(args.kb)
    cfg = default_config()
    tools = discover(
        source_path=cfg.tool_definition_source,
        command=cfg.server_command,
        args=cfg.server_args,
        cwd=cfg.server_cwd,
        env=cfg.server_env,
        prefer="live",
    )
    catalog = _condensed_catalog(tools)
    # One client for both roles: the generator passes its own hot
    # GENERATION_TEMPERATURE per call, so it needs no separate instance.
    # (Cloning via ``LLMClient(**base_llm.__dict__)`` raised TypeError —
    # the dict carries private state the constructor does not accept.)
    base_llm = LLMClient.from_env(env_var=cfg.env_anthropic_api_key)
    generator = TaskGenerator(base_llm, catalog)
    pipeline = BenchmarkPipeline(
        kb,
        config=cfg,
        llm=base_llm,
        allow_replans=args.allow_replans,
        max_replans=args.max_replans,
        log_to_console=not args.quiet,
    )

    accepted = rejected = 0
    session_prompts: list[str] = []
    for i in range(args.count):
        difficulty = (
            args.difficulty
            if args.difficulty != "mixed"
            else _DIFF_CYCLE[i % len(_DIFF_CYCLE)]
        )
        archetype = _pick_archetype(kb, args.category)
        print(
            f"\n=== task {i + 1}/{args.count} "
            f"[{archetype['name']} | readonly | {difficulty}] ==="
        )
        try:
            spec = generator.generate(
                archetype,
                difficulty=difficulty,
                undercovered=_undercovered(kb, catalog),
                avoid=kb.summaries() + session_prompts,
            )
        except (TaskGenError, LLMError) as e:
            # A flaky completion costs this task, not the whole run — the
            # loop is the expensive thing to lose.
            print(f"[taskgen] generation failed: {e}")
            rejected += 1
            continue
        session_prompts.append(spec.task_prompt)
        print(f"[taskgen] prompt: {spec.task_prompt}")
        print(f"[taskgen] expected: {' -> '.join(spec.expected_tools)}")
        for w in spec.warnings:
            print(f"[taskgen][warn] {w}")
        if args.dry_run:
            continue
        result = pipeline.run_spec(spec)
        if result.accepted:
            accepted += 1
            print(
                f"[taskgen] ACCEPTED as {result.entry['id']} "
                f"(match={result.entry['expected_match']})"
            )
        else:
            rejected += 1

    if args.dry_run:
        print(
            f"\n[taskgen] dry run: {args.count} task(s) generated, none executed"
        )
    else:
        print(
            f"\n[taskgen] done: {accepted} accepted, {rejected} rejected; "
            f"KB now has {len(kb.entries)} entries at {kb.path}"
        )
    return 0 if (args.dry_run or accepted > 0) else 1


def cmd_stats(args: argparse.Namespace) -> int:
    kb = KnowledgeBase(args.kb)
    print(json.dumps(kb.stats(), indent=2, ensure_ascii=False))
    return 0


def _pick_archetype(kb: KnowledgeBase, category: str | None) -> dict:
    if category:
        for a in ARCHETYPES:
            if a["name"] == category:
                return a
        sys.exit(
            f"unknown category {category!r}; choose from "
            f"{[a['name'] for a in ARCHETYPES]}"
        )
    counts = kb.category_counts()
    return min(ARCHETYPES, key=lambda a: counts.get(a["name"], 0))


def _undercovered(kb: KnowledgeBase, catalog: list[dict]) -> list[str]:
    """Tools the KB exercises least, to steer generation toward the gaps.

    Originally this returned only tools with ZERO coverage, which went
    silently dead as soon as every tool had one entry — taking the
    diversity signal with it. Ranking by least-covered keeps it useful for
    the whole life of the corpus.
    """
    covered = kb.tool_coverage()
    ranked = sorted(
        (c["tool"] for c in catalog),
        key=lambda t: (covered.get(t, 0), t),
    )
    return ranked[:5]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pkg.finance_taskgen",
        description="Generate finance tasks, verify them via agenticmcpe, and "
                    "grow a task->tool-sequence knowledge base.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("generate", help="Generate and verify tasks into the KB.")
    pg.add_argument("--count", type=int, default=3, help="Tasks to attempt.")
    pg.add_argument("--difficulty", choices=[*DIFFICULTY_STEPS, "mixed"], default="medium")
    pg.add_argument("--category", help="Force one scenario archetype by name.")
    pg.add_argument("--kb", help="Knowledge-base JSON path (default: pkg/finance_taskgen/knowledge_base.json).")
    pg.add_argument("--dry-run", action="store_true",
                    help="Generate task specs only; no execution, no KB writes.")
    pg.add_argument("--allow-replans", action="store_true",
                    help="Accept runs that needed replanning (final plan may not "
                         "be a from-scratch sequence).")
    pg.add_argument("--max-replans", type=int, default=2)
    pg.add_argument("--quiet", action="store_true")
    pg.set_defaults(func=cmd_generate)

    ps = sub.add_parser("stats", help="Show KB size and tool coverage.")
    ps.add_argument("--kb", help="Knowledge-base JSON path.")
    ps.set_defaults(func=cmd_stats)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(f"taskgen error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())