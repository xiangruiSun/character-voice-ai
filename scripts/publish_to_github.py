#!/usr/bin/env python3
"""Create the GitHub repository for this project and push to it, in one command.

    python scripts/publish_to_github.py                     # asks for a token
    python scripts/publish_to_github.py --name my-repo --public
    python scripts/publish_to_github.py --dry-run           # show the plan, do nothing

Standard library only, so it runs wherever the project itself does — no `gh`, no `pip
install`, Windows included. If the GitHub CLI *is* installed and signed in, pass
``--use-gh`` and no token is needed at all.

**The token is never written anywhere.** It is passed to git through an inline credential
helper that reads it from the environment, so it does not end up in `.git/config`, in
your shell history, or in the process list. A fine-grained token with "Administration:
write" and "Contents: write" on new repositories is enough; delete it afterwards.

What it does, in order, and each step is safe to repeat:

1. checks this is a git repository with at least one commit
2. creates the repository on GitHub (an existing one of the same name is reused)
3. points ``origin`` at it, renames the branch to ``main``
4. pushes

It refuses to push a dirty tree. A repository's first commit is the one people read the
history from, and "uncommitted changes, pushed anyway" is a bad first entry.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Reads the token from the environment rather than taking it as an argument, so it
#: never appears in `ps` output or in .git/config.
CREDENTIAL_HELPER = (
    '!f() { echo "username=x-access-token"; echo "password=$CVAI_GH_TOKEN"; }; f'
)


class Failed(RuntimeError):
    """Something went wrong in a way the user needs to read, not a traceback."""


# --------------------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------------------


def git(*args: str, token: str | None = None, check: bool = True) -> str:
    command = ["git", "-C", str(REPO_ROOT)]
    environment = dict(os.environ)
    if token:
        command += ["-c", f"credential.helper={CREDENTIAL_HELPER}"]
        environment["CVAI_GH_TOKEN"] = token
    command += list(args)

    result = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise Failed(f"git {' '.join(args)} failed:\n{detail}")
    return result.stdout.strip()


def preflight() -> None:
    if not (REPO_ROOT / ".git").exists():
        raise Failed(
            f"{REPO_ROOT} is not a git repository. If you extracted this from an "
            "archive, make sure the .git directory came with it."
        )
    if not git("rev-parse", "--verify", "HEAD", check=False):
        raise Failed("this repository has no commits yet; there is nothing to push.")
    if git("status", "--porcelain"):
        raise Failed(
            "you have uncommitted changes. Commit or stash them first — the first push "
            "is the history everyone else reads."
        )


# --------------------------------------------------------------------------------------
# GitHub
# --------------------------------------------------------------------------------------


def api(path: str, token: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "cvai-publish",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        if error.code == 401:
            raise Failed(
                "GitHub rejected the token (401). Check that it has not expired and "
                "that it grants Administration: write and Contents: write."
            ) from error
        raise Failed(f"GitHub returned {error.code} for {path}:\n{body}") from error
    except urllib.error.URLError as error:
        raise Failed(f"could not reach GitHub: {error.reason}") from error


def create_repository(
    token: str, name: str, *, private: bool, description: str
) -> dict:
    account = api("/user", token)
    owner = account["login"]

    existing = _existing(token, owner, name)
    if existing is not None:
        print(f"  repository {owner}/{name} already exists — reusing it")
        return existing

    print(f"  creating {'private' if private else 'public'} repository {owner}/{name}")
    return api(
        "/user/repos",
        token,
        {
            "name": name,
            "private": private,
            "description": description,
            # No auto-init: an initial commit created by GitHub would have to be merged
            # with ours before the first push, for no benefit.
            "auto_init": False,
            "has_wiki": False,
        },
    )


def _existing(token: str, owner: str, name: str) -> dict | None:
    try:
        return api(f"/repos/{owner}/{name}", token)
    except Failed:
        return None


# --------------------------------------------------------------------------------------
# The GitHub CLI path
# --------------------------------------------------------------------------------------


def publish_with_gh(name: str, private: bool, description: str) -> str:
    visibility = "--private" if private else "--public"
    print("  handing over to the GitHub CLI")
    result = subprocess.run(
        [
            "gh", "repo", "create", name, visibility,
            "--source", str(REPO_ROOT), "--remote", "origin", "--push",
            "--description", description,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise Failed((result.stderr or result.stdout).strip())
    url = next(
        (line.strip() for line in result.stdout.splitlines() if line.startswith("http")),
        "",
    )
    return url or f"https://github.com/<you>/{name}"


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", default="character-voice-ai", help="repository name")
    parser.add_argument(
        "--public", action="store_true", help="create it public (default: private)"
    )
    parser.add_argument("--branch", default="main", help="branch to push (default: main)")
    parser.add_argument(
        "--description",
        default="Character-specific Chinese voice AI — not generic TTS, not zero-shot cloning.",
    )
    parser.add_argument(
        "--token",
        help="GitHub token. Prefer the GITHUB_TOKEN environment variable, or let it "
        "prompt — an argument is visible in your shell history.",
    )
    parser.add_argument(
        "--use-gh", action="store_true", help="use the GitHub CLI instead of a token"
    )
    parser.add_argument("--dry-run", action="store_true", help="say what it would do")
    args = parser.parse_args(argv)

    try:
        return run(args)
    except Failed as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print()
        return 130


def run(args: argparse.Namespace) -> int:
    print(f"Publishing {REPO_ROOT.name}")
    preflight()

    branch = args.branch
    current = git("rev-parse", "--abbrev-ref", "HEAD")
    commits = git("rev-list", "--count", "HEAD")
    print(f"  {commits} commits on '{current}', pushing as '{branch}'")

    if args.dry_run:
        print("\n(dry run — nothing was created or pushed)")
        return 0

    if args.use_gh:
        url = publish_with_gh(args.name, not args.public, args.description)
        print(f"\nDone: {url}")
        return 0

    token = args.token or os.environ.get("GITHUB_TOKEN") or getpass.getpass(
        "GitHub token (input hidden): "
    )
    if not token.strip():
        raise Failed("no token given.")
    token = token.strip()

    repository = create_repository(
        token, args.name, private=not args.public, description=args.description
    )
    url = repository["clone_url"]
    html = repository["html_url"]

    if current != branch:
        git("branch", "-M", branch)
    if git("remote", check=False):
        git("remote", "remove", "origin", check=False)
    git("remote", "add", "origin", url)

    print("  pushing")
    git("push", "-u", "origin", branch, token=token)

    print(f"\nDone: {html}")
    print("Delete the token now if you made it for this — it is not needed again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
