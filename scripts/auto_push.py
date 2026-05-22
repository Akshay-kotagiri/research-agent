"""
Auto-commit and push watcher.

Watches src/, tests/, and .github/ for changes to .py and .yml files.
On save, runs: git add . -> git commit -> git push origin master.
A 2-second debounce prevents duplicate commits from rapid saves.

Usage:
    python scripts/auto_push.py
    python scripts/auto_push.py --branch main   # push to a different branch
"""
from __future__ import annotations

import argparse
import subprocess
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).resolve().parent.parent
WATCH_DIRS = [ROOT / "src", ROOT / "tests", ROOT / ".github"]
DEBOUNCE_SECONDS = 2.0


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def _remove_index_lock() -> None:
    lock = ROOT / ".git" / "index.lock"
    if lock.exists():
        lock.unlink()
        print("[auto-push] Removed stale .git/index.lock")


def _git_add_commit_push(path: Path, branch: str) -> None:
    rel = path.relative_to(ROOT)

    _remove_index_lock()

    add = _run(["git", "add", "."])
    if add.returncode != 0:
        print(f"[auto-push] git add failed:\n{add.stderr.strip()}")
        return

    commit = _run(["git", "commit", "-m", f"auto: save {rel.name}"])
    output = (commit.stdout + commit.stderr).strip()
    if "nothing to commit" in output:
        print(f"[auto-push] {rel.name} — no changes staged, skipping.")
        return
    if commit.returncode != 0:
        print(f"[auto-push] git commit failed:\n{output}")
        return

    push = _run(["git", "push", "origin", branch])
    if push.returncode != 0:
        print(f"[auto-push] git push failed:\n{push.stderr.strip()}")
        return

    print(f"[auto-push] ✓ {rel}  →  pushed to {branch}")


class _DebouncedHandler(FileSystemEventHandler):
    def __init__(self, branch: str) -> None:
        super().__init__()
        self._branch = branch
        self._timers: dict[Path, threading.Timer] = {}
        self._lock = threading.Lock()

    def _schedule(self, path: Path) -> None:
        with self._lock:
            existing = self._timers.pop(path, None)
            if existing:
                existing.cancel()
            timer = threading.Timer(
                DEBOUNCE_SECONDS,
                _git_add_commit_push,
                args=(path, self._branch),
            )
            self._timers[path] = timer
            timer.start()

    def _handle(self, event) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix in {".py", ".yml"} and not path.name.startswith("."):
            print(f"[auto-push] Change detected: {path.relative_to(ROOT)}")
            self._schedule(path)

    on_modified = _handle
    on_created = _handle


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto git-commit and push on .py file save.")
    parser.add_argument("--branch", default="master", help="Branch to push to (default: master)")
    args = parser.parse_args()

    handler = _DebouncedHandler(branch=args.branch)
    observer = Observer()

    for watch_dir in WATCH_DIRS:
        if watch_dir.exists():
            observer.schedule(handler, str(watch_dir), recursive=True)
            print(f"[auto-push] Watching {watch_dir.relative_to(ROOT)}/")
        else:
            print(f"[auto-push] Skipping {watch_dir.relative_to(ROOT)}/ (not found)")

    print(f"[auto-push] Branch: {args.branch}  |  Debounce: {DEBOUNCE_SECONDS}s  |  Ctrl+C to stop\n")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[auto-push] Stopping watcher.")
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()
