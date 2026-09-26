"""CLI for the taskgen benchmark builder.

    python -m pkg.taskgen generate --count 3                        # full loop
    python -m pkg.taskgen generate --count 2 --dry-run              # prompts only
    python -m pkg.taskgen stats                                     # KB coverage
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

from pkg.agenticmcpe.catalog import condense_catalog, load_catalog
from pkg.agenticmcpe.config import ConfigError, LLMClient, Settings

from .generator import (ARCHETYPES, DIFFICULTY_STEPS, GENERATION_TEMPERATURE,
                        TaskGenError, TaskGenerator)
from .kb import KnowledgeBase
from .pipeline import BenchmarkPipeline

_DIFF_CYCLE = ["easy", "medium", "hard"]

# --style-benchmark: which benchmark prompt files feed each archetype as style
# exemplars, so bred entries land NEAR the benchmark's task distribution
# (retrieval similarity is lexical — register and shape matter).
_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "browser_automation"
_STYLE_PREFIX = {
    "flight-price-lookup": "playwright_booking_task_",
    "maps-route": "playwright_google_map_task_",
    "model-hub-lookup": "playwright_huggingface_task_",
    "paper-lookup": "playwright_paper_task_",
    "sports-schedule": "playwright_sports_task_",
}


def _benchmark_prompts(archetype_name: str) -> list[str]:
    prefix = _STYLE_PREFIX.get(archetype_name)
    if not prefix:
        return []
    out = []
    for p in sorted(_BENCHMARK_DIR.glob(prefix + "*.txt")):
        _, sep, prompt = p.read_text(encoding="utf-8").partition("PROMPT:")
        if sep:
            out.append(prompt.strip())
    return out


def cmd_generate(args: argparse.Namespace) -> int:
    kb = KnowledgeBase(args.kb)
    gen_settings = Settings.load(provider=args.provider,
                                 run_id=time.strftime("taskgen-%Y%m%d-%H%M%S"))
    catalog = condense_catalog(load_catalog(gen_settings))
    # Run the generator hotter than the default temperature-0 client so tasks
    # vary; the pipeline's planner/verifier still use the cold default settings.
    gen_llm = LLMClient(replace(gen_settings.llm, temperature=GENERATION_TEMPERATURE))
    generator = TaskGenerator(gen_llm, catalog)
    pipeline = BenchmarkPipeline(
        kb,
        provider=args.provider,
        allow_replans=args.allow_replans,
        max_replans=args.max_replans,
        log_to_console=not args.quiet,
    )

    accepted = rejected = 0
    session_prompts: list[str] = []
    for i in range(args.count):
        difficulty = (args.difficulty if args.difficulty != "mixed"
                      else _DIFF_CYCLE[i % len(_DIFF_CYCLE)])
        archetype = _pick_archetype(kb, args.category)
        print(f"\n=== task {i + 1}/{args.count} "
              f"[{archetype['name']} | {difficulty}] ===")
        exemplar_pool = (_benchmark_prompts(archetype["name"])
                         if args.style_benchmark else [])
        try:
            spec = generator.generate(
                archetype,
                difficulty=difficulty,
                undercovered=_undercovered(kb, catalog),
                # Benchmark prompts join the avoid list so a bred task can
                # never be a copy of the very tasks it emulates.
                avoid=kb.summaries() + session_prompts + exemplar_pool,
                style_exemplars=(random.sample(exemplar_pool,
                                               min(3, len(exemplar_pool)))
                                 if exemplar_pool else None),
            )
        except TaskGenError as e:
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
            print(f"[taskgen] ACCEPTED as {result.entry['id']} "
                  f"(match={result.entry['expected_match']})")
        else:
            rejected += 1

    if args.dry_run:
        print(f"\n[taskgen] dry run: {args.count} task(s) generated, none executed")
    else:
        print(f"\n[taskgen] done: {accepted} accepted, {rejected} rejected; "
              f"KB now has {len(kb.entries)} entries at {kb.path}")
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
        sys.exit(f"unknown category {category!r}; choose from "
                 f"{[a['name'] for a in ARCHETYPES]}")
    counts = kb.category_counts()
    return min(ARCHETYPES, key=lambda a: counts.get(a["name"], 0))


def _undercovered(kb: KnowledgeBase, catalog: list[dict]) -> list[str]:
    from .generator import UNSUPPORTED_TOOLS
    covered = kb.tool_coverage()
    pool = [c["tool"] for c in catalog
            if c["tool"] not in UNSUPPORTED_TOOLS
            and covered.get(c["tool"], 0) == 0]
    return pool[:15]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pkg.taskgen",
        description="Generate browser tasks, verify them via agenticmcpe, and "
                    "grow a task->tool-sequence knowledge base.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("generate", help="Generate and verify tasks into the KB.")
    pg.add_argument("--count", type=int, default=3, help="Tasks to attempt.")
    pg.add_argument("--difficulty", choices=[*DIFFICULTY_STEPS, "mixed"],
                    default="medium")
    pg.add_argument("--category", help="Force one scenario archetype by name.")
    pg.add_argument("--provider", help="LLM provider (deepseek|openai|gemini|anthropic|"
                                       "minimax|qwen|glm|kimi|grok|xiaomi|custom).")
    pg.add_argument("--kb", help="Knowledge-base JSON path (default: pkg/taskgen/knowledge_base.json).")
    pg.add_argument("--dry-run", action="store_true",
                    help="Generate task specs only; no execution, no KB writes.")
    pg.add_argument("--allow-replans", action="store_true",
                    help="Accept runs that needed replanning (final plan may not "
                         "be a from-scratch sequence).")
    pg.add_argument("--style-benchmark", action="store_true",
                    help="Seed generation with the benchmark prompts of the "
                         "archetype's category as style exemplars, so entries "
                         "land near the benchmark's task distribution.")
    pg.add_argument("--max-replans", type=int, default=2)
    pg.add_argument("--quiet", action="store_true")
    pg.set_defaults(func=cmd_generate)

    ps = sub.add_parser("stats", help="Show KB size and tool coverage.")
    ps.add_argument("--kb", help="Knowledge-base JSON path.")
    ps.set_defaults(func=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
