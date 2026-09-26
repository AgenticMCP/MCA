"""Delete the GitHub repos an agenticmcpe batch run created.

Teardown for the 28-task batch (and any run that leaves repos behind). It works
off the **live account**, not the run traces: it lists every repo the
authenticated user owns and deletes all of them except a fixed keep-forever set
(PRESERVE_REPOS).

Why not read the run traces? Because they drift from the account. Runs get
re-run (overwriting trace.json), replans drop their attempt archives, and a
repo's creation can survive only in the task prompt — so a trace scan both
misses repos that still exist (e.g. a fork whose fork_repository step wasn't
archived) and lists repos that were already deleted. The account itself is the
only reliable answer to "what did we create that's still here", so that's what
this reads (GET /user/repos?affiliation=owner).

Safe by default: it prints the targets and exits (DRY RUN). Pass ``--yes`` to
actually delete. Two guards keep it from touching anything you want to keep:

  * owner scope   — only repos *owned* by the authenticated user are listed;
  * preserve list — PRESERVE_REPOS (extend per-run with ``--preserve a,b``) is
                    always skipped.

Because the policy is "everything owned except the keep-list", always eyeball
the dry-run before ``--yes``: any repo on the account that is not in the
preserve list is a target.

Auth mirrors ``config.py``: ``GITHUB_TOKENS`` (comma) →
``GITHUB_PERSONAL_ACCESS_TOKEN`` → ``GITHUB_TOKEN``, read from the environment
or ``pkg/agenticmcpe/.env``. Deleting a repo needs a token with the
``delete_repo`` scope (planning/execution does not) — a 403 is reported per
repo and skipped; pass ``--token`` to supply a stronger one.

    # dry run — list every owned repo that WOULD be deleted
    pkg/venv/bin/python pkg/agenticmcpe/run/cleanup_repos.py

    # delete them for real
    pkg/venv/bin/python pkg/agenticmcpe/run/cleanup_repos.py --yes

    # protect an extra repo just for this run
    pkg/venv/bin/python pkg/agenticmcpe/run/cleanup_repos.py --preserve my-keeper --yes

Stdlib only — runs under any python3, no venv or package import required.
Modifies nothing in the package.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent

# Repos owned by the test account that no task creates and must NEVER be
# deleted. This is the sole guard for the account-wide sweep, so keep it
# current: any repo you want to keep on the account must be listed here (or
# passed via --preserve).
PRESERVE_REPOS = {
    "build-your-own-x", "claude-code", "EasyR1", "harmony",
    "mcpmark-cicd", "missing-semester",
}


def load_dotenv() -> None:
    """Populate os.environ from ``.env`` without overriding set vars. Mirrors
    config.load_dotenv: CWD/.env then the package .env; ``#`` comments,
    ``export KEY=...`` and single/double quotes."""
    for p in (Path.cwd() / ".env", PKG_DIR / ".env"):
        if not p.is_file():
            continue
        for raw in p.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, sep, val = line.partition("=")
            if not sep:
                continue
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            os.environ.setdefault(key, val)


def resolve_token(cli_token: str | None) -> str | None:
    """First token by config.py precedence: --token, GITHUB_TOKENS (comma),
    GITHUB_PERSONAL_ACCESS_TOKEN, GITHUB_TOKEN."""
    if cli_token:
        return cli_token
    for t in os.environ.get("GITHUB_TOKENS", "").split(","):
        if t.strip():
            return t.strip()
    for single in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_TOKEN"):
        v = os.environ.get(single)
        if v:
            return v
    return None


def gh_api(method: str, url: str, token: str):
    req = urllib.request.Request(
        url, method=method,
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"})
    return urllib.request.urlopen(req, timeout=30)


def list_owned_repos(token: str) -> list[dict]:
    """Every repo the authenticated user owns (paginated). affiliation=owner
    excludes org/collaborator repos, so only the account's own repos appear."""
    repos: list[dict] = []
    page = 1
    while True:
        url = ("https://api.github.com/user/repos"
               f"?affiliation=owner&per_page=100&page={page}&sort=created")
        batch = json.load(gh_api("GET", url, token))
        repos += batch
        if len(batch) < 100:
            return repos
        page += 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Delete every repo the account owns except the preserve "
                    "list (dry run unless --yes).")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="actually delete (default: dry run — print only)")
    ap.add_argument("--preserve", default="",
                    help="comma-separated extra repo names to keep this run")
    ap.add_argument("--token", help="GitHub token override (needs delete_repo scope)")
    args = ap.parse_args(argv)

    load_dotenv()
    token = resolve_token(args.token)
    if not token:
        print("error: no GitHub token — set GITHUB_PERSONAL_ACCESS_TOKEN / "
              "GITHUB_TOKEN / GITHUB_TOKENS in env or pkg/agenticmcpe/.env, "
              "or pass --token")
        return 2
    try:
        login = json.load(gh_api("GET", "https://api.github.com/user", token))["login"]
        owned = list_owned_repos(token)
    except Exception as e:
        print(f"error: GitHub API call failed: {e!r}")
        return 2

    preserve = PRESERVE_REPOS | {p.strip() for p in args.preserve.split(",") if p.strip()}
    targets, preserved = [], []
    for r in sorted(owned, key=lambda r: r["name"].lower()):
        (preserved if r["name"] in preserve else targets).append(r)

    print(f"account {login}: {len(owned)} owned repo(s)   "
          f"to delete: {len(targets)}   preserved: {len(preserved)}")
    for r in preserved:
        print(f"  - keep (preserve list): {r['full_name']}")
    if not targets:
        print("nothing to delete.")
        return 0

    kind = lambda r: "fork" if r.get("fork") else "repo"
    if not args.yes:
        print("\nDRY RUN — would delete:")
        for r in targets:
            print(f"  - {r['full_name']}  [{kind(r)}, created {r['created_at']}]")
        print(f"\nRe-run with --yes to delete these {len(targets)} repo(s).")
        return 0

    print("\ndeleting:")
    deleted = 0
    for r in targets:
        try:
            gh_api("DELETE", "https://api.github.com/repos/" + r["full_name"], token)
            print(f"  deleted {r['full_name']}  [{kind(r)}]")
            deleted += 1
        except urllib.error.HTTPError as e:
            hint = ("  (token needs delete_repo scope)" if e.code == 403 else
                    "  (already gone)" if e.code == 404 else "")
            print(f"  skip {r['full_name']}: HTTP {e.code} {e.reason}{hint}")
        except Exception as e:
            print(f"  skip {r['full_name']}: {e!r}")
    print(f"\ndone: {deleted}/{len(targets)} deleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
