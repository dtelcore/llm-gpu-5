"""
gitpush.py - One-shot helper that runs the same git sequence used throughout this
project's sessions to publish local changes to origin/main:

    1. git status --short          (show what is about to be committed)
    2. git add -A                  (stage everything)
    3. git commit -F <message>     (commit using a message file, PowerShell-safe)
    4. git push origin <branch>    (publish)

Usage:
    python gitpush.py "commit message here"
    python gitpush.py              # prompts for a commit message interactively

If there is nothing to commit, the script reports that and exits cleanly without
calling git commit/push.
"""

import os
import subprocess
import sys
import tempfile

DEFAULT_BRANCH = "main"


def run(cmd, **kwargs):
    """Run a command, streaming output, and return the completed process."""
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False, text=True, **kwargs)


def git_status_short():
    result = run(["git", "status", "--short"], capture_output=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.stdout


def current_branch():
    result = run(["git", "branch", "--show-current"], capture_output=True)
    return result.stdout.strip() or DEFAULT_BRANCH


def has_staged_or_unstaged_changes():
    result = subprocess.run(["git", "status", "--porcelain"], check=False,
                             text=True, capture_output=True)
    return bool(result.stdout.strip())


def commit_with_message_file(message: str) -> bool:
    """Writes the commit message to a temp file and commits via -F, avoiding
    PowerShell heredoc/quoting issues entirely."""
    fd, path = tempfile.mkstemp(prefix="gitpush_commit_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(message.strip() + "\n")
        result = run(["git", "commit", "-F", path])
        return result.returncode == 0
    finally:
        if os.path.exists(path):
            os.remove(path)


def main():
    print("=" * 70)
    print("GITPUSH: status -> add -> commit -> push")
    print("=" * 70)

    status_output = git_status_short()

    if not has_staged_or_unstaged_changes():
        print("\nNothing to commit; working tree is clean. Skipping commit/push.")
        return 0

    run(["git", "add", "-A"])

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
    else:
        print("\nFiles to be committed:")
        print(status_output)
        message = input("Commit message: ").strip()
        if not message:
            print("Empty commit message; aborting before commit/push.")
            return 1

    if not commit_with_message_file(message):
        print("\ngit commit failed; aborting before push.")
        return 1

    branch = current_branch()
    push_result = run(["git", "push", "origin", branch])
    if push_result.returncode != 0:
        print(f"\ngit push to origin/{branch} failed.")
        return 1

    print(f"\nDone: committed and pushed to origin/{branch}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
