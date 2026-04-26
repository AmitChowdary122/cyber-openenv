"""Delete obsolete Colab/legacy files from local disk AND the HF Space remote.

`hf upload` only adds/updates files; it never deletes from the remote, so a
mirror upload after deleting locally still leaves the files on the Space.
This script handles both sides cleanly:

  1. Delete the obsolete files/folders from the local working tree
  2. Stage their deletion in git (so the next push removes them from GitHub)
  3. Delete the same paths from the HF Space remote via HfApi.delete_file/folder

Run from the repo root:
    python scripts/cleanup_obsolete.py

Requires: huggingface_hub, and `hf auth login` already done.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_REPO = "amit51/cybersoc-arena"

# (path_in_repo, kind) where kind is "file" or "folder"
OBSOLETE = [
    # ONLY the old GRPO notebook -- DO NOT delete the notebooks/ folder, the
    # current Colab demo (notebooks/CyberSOC_Arena_demo.ipynb) lives there.
    ("notebooks/CyberSOC_Arena_GRPO.ipynb", "file"),
    ("scripts/run_hf_job.sh", "file"),
    ("scripts/run_hf_job_a100.sh", "file"),  # renamed to run_hf_job_l40s.sh
    ("PRESENTATION.md", "file"),
    ("SUBMIT.md", "file"),
]


def delete_local(path_in_repo: str, kind: str) -> None:
    full = os.path.join(REPO_ROOT, path_in_repo)
    if not os.path.exists(full):
        print(f"  [local] {path_in_repo}  -- already gone")
        return
    if kind == "folder":
        shutil.rmtree(full)
    else:
        os.remove(full)
    print(f"  [local] {path_in_repo}  -- deleted")


def git_rm(path_in_repo: str) -> None:
    """Stage the deletion in git. Ignores 'did not match' if already untracked."""
    try:
        subprocess.run(
            ["git", "rm", "-rf", "--ignore-unmatch", path_in_repo],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        print(f"  [git]   {path_in_repo}  -- staged for deletion")
    except subprocess.CalledProcessError as e:
        print(f"  [git]   {path_in_repo}  -- WARN: {e.stderr.decode().strip()}")


def delete_remote(api: HfApi, path_in_repo: str, kind: str) -> None:
    try:
        if kind == "folder":
            api.delete_folder(
                path_in_repo=path_in_repo,
                repo_id=SPACE_REPO,
                repo_type="space",
                commit_message=f"chore: drop obsolete {path_in_repo}/",
            )
        else:
            api.delete_file(
                path_in_repo=path_in_repo,
                repo_id=SPACE_REPO,
                repo_type="space",
                commit_message=f"chore: drop obsolete {path_in_repo}",
            )
        print(f"  [space] {path_in_repo}  -- deleted from {SPACE_REPO}")
    except EntryNotFoundError:
        print(f"  [space] {path_in_repo}  -- not on Space (skip)")
    except HfHubHTTPError as e:
        msg = str(e).splitlines()[0] if str(e) else "unknown HTTP error"
        if "404" in msg or "not found" in msg.lower():
            print(f"  [space] {path_in_repo}  -- not on Space (skip)")
        else:
            print(f"  [space] {path_in_repo}  -- WARN: {msg}")


def main() -> int:
    api = HfApi()

    print("== Local + git deletions ==")
    for path, kind in OBSOLETE:
        delete_local(path, kind)
        git_rm(path)

    print("\n== HF Space remote deletions ==")
    for path, kind in OBSOLETE:
        delete_remote(api, path, kind)

    print("\n== DONE ==")
    print("Now finish the GitHub side:")
    print("  git commit -m 'drop: Colab notebook, T4 launcher, PRESENTATION.md, SUBMIT.md'")
    print("  git push origin main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
