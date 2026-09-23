"""The one-command GitHub publisher.

Most of it is network, which is not worth mocking end to end — but three things are worth
holding, because each is a mistake that survives into a public repository:

* **The token must not be written anywhere.** Embedding it in the remote URL is the usual
  way to do this, and it leaves the credential sitting in `.git/config` on the developer's
  disk and in every subsequent `git remote -v`.
* **A dirty tree must not be pushed.** The first push is the history everyone else reads.
* **Running it twice must not hurt.** People re-run a failed publish immediately, and an
  "already exists" error that looks like a crash sends them editing things by hand.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

publish = pytest.importorskip("scripts.publish_to_github", reason="script import")


@pytest.fixture(autouse=True)
def _in_a_temp_repo(tmp_path: Path, monkeypatch):
    """Point the script at a throwaway repository, never the real one."""
    monkeypatch.setattr(publish, "REPO_ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    return tmp_path


def commit(root: Path, name: str = "a.txt") -> None:
    (root / name).write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "first"], check=True)


# --------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------


def test_a_clean_repository_with_commits_passes(_in_a_temp_repo):
    commit(_in_a_temp_repo)
    publish.preflight()


def test_a_repository_with_no_commits_is_refused(_in_a_temp_repo):
    with pytest.raises(publish.Failed, match="no commits"):
        publish.preflight()


def test_uncommitted_changes_are_refused(_in_a_temp_repo):
    commit(_in_a_temp_repo)
    (_in_a_temp_repo / "b.txt").write_text("later\n", encoding="utf-8")
    with pytest.raises(publish.Failed, match="uncommitted"):
        publish.preflight()


def test_a_missing_git_directory_says_what_went_wrong(tmp_path, monkeypatch):
    """The likely cause is an archive extracted without its .git, so say that."""
    monkeypatch.setattr(publish, "REPO_ROOT", tmp_path / "nowhere")
    with pytest.raises(publish.Failed, match="not a git repository"):
        publish.preflight()


# --------------------------------------------------------------------------------------
# The token
# --------------------------------------------------------------------------------------


def test_the_token_is_passed_by_environment_not_in_the_command():
    """In argv it is visible in `ps` to every other process on the machine."""
    assert "$CVAI_GH_TOKEN" in publish.CREDENTIAL_HELPER
    assert "x-access-token" in publish.CREDENTIAL_HELPER


def test_git_only_configures_a_helper_when_there_is_a_token(_in_a_temp_repo, monkeypatch):
    seen: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        seen.append(list(command))
        return Result()

    monkeypatch.setattr(publish.subprocess, "run", fake_run)
    publish.git("status")
    publish.git("push", token="secret-token")

    assert not any("credential.helper" in part for part in seen[0])
    assert any("credential.helper" in part for part in seen[1])
    # And the token itself is nowhere in the arguments.
    assert not any("secret-token" in part for part in seen[1])


def test_the_token_never_reaches_the_repository_config(_in_a_temp_repo, monkeypatch):
    commit(_in_a_temp_repo)
    monkeypatch.setenv("CVAI_GH_TOKEN", "secret-token")
    publish.git("remote", "add", "origin", "https://github.com/someone/repo.git")

    config = (_in_a_temp_repo / ".git" / "config").read_text(encoding="utf-8")
    assert "secret-token" not in config
    assert "https://github.com/someone/repo.git" in config


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------


def test_it_defaults_to_a_private_repository(capsys, _in_a_temp_repo):
    """A character voice project may involve audio whose rights are unsettled; public
    is a choice the person makes, not a default they discover."""
    commit(_in_a_temp_repo)
    assert publish.main(["--dry-run"]) == 0
    assert "nothing was created" in capsys.readouterr().out


def test_dry_run_touches_nothing(_in_a_temp_repo, monkeypatch):
    commit(_in_a_temp_repo)
    monkeypatch.setattr(
        publish, "create_repository", lambda *a, **k: pytest.fail("created a repo")
    )
    assert publish.main(["--dry-run"]) == 0
    assert "origin" not in publish.git("remote")


def test_a_failure_is_a_message_not_a_traceback(_in_a_temp_repo, capsys):
    assert publish.main(["--dry-run"]) == 1
    assert "error:" in capsys.readouterr().err
