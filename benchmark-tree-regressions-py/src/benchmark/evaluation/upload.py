"""Upload benchmark results to a git repository."""

import os
import platform
import subprocess
import datetime
from pathlib import Path
from typing import Optional

from ..config import RESULTS_DIR


def _run_git_command(args: list, cwd: str | Path) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def is_git_repo(path: str | Path) -> bool:
    """Check if a directory is a git repository."""
    rc, _, _ = _run_git_command(["rev-parse", "--git-dir"], cwd=path)
    return rc == 0


def init_repo(path: str | Path, remote_url: Optional[str] = None) -> bool:
    """
    Initialize a git repository.

    Parameters
    ----------
    path : str or Path
        Directory to initialize as git repo.
    remote_url : str, optional
        URL of remote repository (e.g., GitHub URL).

    Returns
    -------
    bool
        True if successful.
    """
    path = Path(path)

    # Initialize repo
    rc, stdout, stderr = _run_git_command(["init"], cwd=path)
    if rc != 0 and "already exists" not in stderr.lower():
        raise RuntimeError(f"Failed to init git repo: {stderr}")

    # Add remote if provided
    if remote_url:
        # Check if remote already exists
        rc, stdout, _ = _run_git_command(["remote", "-v"], cwd=path)
        if "origin" not in stdout:
            rc, _, stderr = _run_git_command(
                ["remote", "add", "origin", remote_url], cwd=path
            )
            if rc != 0:
                raise RuntimeError(f"Failed to add remote: {stderr}")
        else:
            rc, _, stderr = _run_git_command(
                ["remote", "set-url", "origin", remote_url], cwd=path
            )
            if rc != 0:
                raise RuntimeError(f"Failed to set remote URL: {stderr}")

    return True


def get_user_info() -> tuple[str, str]:
    """Get user name and email from git config, or return defaults."""
    name = subprocess.run(
        ["git", "config", "user.name"], capture_output=True, text=True
    ).stdout.strip() or os.environ.get("USER", "anonymous")

    email = subprocess.run(
        ["git", "config", "user.email"], capture_output=True, text=True
    ).stdout.strip() or f"{name}@localhost"

    return name, email


def upload_results(
    repo_path: str | Path,
    results_file: str | Path = "benchmark_synthetic.csv",
    remote: str = "origin",
    branch: str = "main",
    message: Optional[str] = None,
) -> dict:
    """
    Upload benchmark results to a git repository.

    Parameters
    ----------
    repo_path : str or Path
        Path to the git repository root.
    results_file : str or Path
        Name of the results file to upload (relative to RESULTS_DIR).
    remote : str
        Name of the remote to push to (default: "origin").
    branch : str
        Name of the branch to push to (default: "main").
    message : str, optional
        Custom commit message. If None, generates a default message.

    Returns
    -------
    dict
        Status information about the upload.
    """
    repo_path = Path(repo_path)
    results_path = RESULTS_DIR / results_file

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    # Ensure repo is git repo
    if not is_git_repo(repo_path):
        raise RuntimeError(f"Not a git repository: {repo_path}")

    # Check if results file is tracked
    rc, _, _ = _run_git_command(
        ["ls-files", "--error-unmatch", str(results_path.relative_to(repo_path))],
        cwd=repo_path,
    )
    is_tracked = rc == 0

    # Stage the file
    rel_path = results_path.relative_to(repo_path)
    rc, _, stderr = _run_git_command(["add", str(rel_path)], cwd=repo_path)
    if rc != 0:
        raise RuntimeError(f"Failed to stage file: {stderr}")

    # Check if there are changes to commit
    rc, stdout, _ = _run_git_command(["status", "--porcelain"], cwd=repo_path)
    if not stdout.strip():
        return {
            "success": True,
            "message": "No changes to commit",
            "committed": False,
        }

    # Generate commit message
    if message is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system = platform.system()
        python_version = platform.python_version()
        hostname = platform.node()

        message = f"""Auto-upload benchmark results ({timestamp})

System: {system} ({hostname})
Python: {python_version}
Results: {results_file}
Branch: {branch}"""

    # Commit
    # Set user info if not configured
    name, email = get_user_info()
    _run_git_command(["config", "user.name", name], cwd=repo_path)
    _run_git_command(["config", "user.email", email], cwd=repo_path)

    rc, _, stderr = _run_git_command(["commit", "-m", message], cwd=repo_path)
    if rc != 0:
        raise RuntimeError(f"Failed to commit: {stderr}")

    committed = True

    # Push to remote
    rc, stdout, stderr = _run_git_command(
        ["push", "-u", remote, branch], cwd=repo_path
    )
    if rc != 0:
        raise RuntimeError(
            f"Failed to push to {remote}/{branch}: {stderr}\n"
            "Please ensure you have push permissions and the remote is configured."
        )

    return {
        "success": True,
        "message": "Results uploaded successfully",
        "committed": committed,
        "branch": branch,
        "remote": remote,
    }


def upload_all_results(
    repo_path: str | Path,
    remote: str = "origin",
    branch: str = "main",
) -> dict:
    """
    Upload all benchmark results (synthetic and real) to git repository.

    Parameters
    ----------
    repo_path : str or Path
        Path to the git repository root.
    remote : str
        Name of the remote to push to.
    branch : str
        Name of the branch to push to.

    Returns
    -------
    dict
        Status information about the uploads.
    """
    results_files = ["benchmark_synthetic.csv", "benchmark_real.csv"]
    results = {}

    for results_file in results_files:
        file_path = RESULTS_DIR / results_file
        if file_path.exists():
            try:
                results[results_file] = upload_results(
                    repo_path=repo_path,
                    results_file=results_file,
                    remote=remote,
                    branch=branch,
                )
            except Exception as e:
                results[results_file] = {"success": False, "error": str(e)}
        else:
            results[results_file] = {"success": False, "error": "File not found"}

    return results


def setup_and_upload(
    repo_path: str | Path,
    remote_url: Optional[str] = None,
    results_file: str = "benchmark_synthetic.csv",
    remote: str = "origin",
    branch: str = "main",
) -> dict:
    """
    Initialize git repo (if needed) and upload results.

    Parameters
    ----------
    repo_path : str or Path
        Path to the git repository root.
    remote_url : str, optional
        URL of remote repository.
    results_file : str
        Name of the results file to upload.
    remote : str
        Name of the remote.
    branch : str
        Name of the branch.

    Returns
    -------
    dict
        Status information.
    """
    repo_path = Path(repo_path)

    if not is_git_repo(repo_path):
        init_repo(repo_path, remote_url)

    return upload_results(
        repo_path=repo_path,
        results_file=results_file,
        remote=remote,
        branch=branch,
    )
