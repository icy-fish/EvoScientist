"""Custom backends for EvoScientist agent."""

import os
import re
import uuid
from pathlib import Path

from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.backends.filesystem import WriteResult, EditResult
from deepagents.backends.protocol import (
    BackendProtocol,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)

# System path prefixes that should never appear in virtual paths.
# If the agent hallucinates an absolute system path, we block it.
_SYSTEM_PATH_PREFIXES = (
    "/Users/", "/home/", "/tmp/", "/var/", "/etc/",
    "/opt/", "/usr/", "/bin/", "/sbin/", "/dev/",
    "/proc/", "/sys/", "/root/",
)

# Dangerous patterns that could escape the workspace
BLOCKED_PATTERNS = [
    r'\.\.',              # ../ directory traversal
    r'~/',                # home directory
    r'\bcd\s+/',          # cd to absolute path
    r'\brm\s+-rf\s+/',    # rm -rf with absolute path
]

# Dangerous commands that should never be executed
BLOCKED_COMMANDS = [
    'sudo',
    'chmod',
    'chown',
    'mkfs',
    'dd',
    'shutdown',
    'reboot',
]


def validate_command(command: str) -> str | None:
    """
    Validate a shell command for safety.

    Returns:
        None if command is safe, error message string if blocked.
    """
    # Check for directory traversal and dangerous patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return (
                f"Command blocked: contains forbidden pattern '{pattern}'. "
                f"All commands must operate within the workspace directory. "
                f"Use relative paths (e.g., './file.py') instead."
            )

    # Check for dangerous commands
    for cmd in BLOCKED_COMMANDS:
        if re.search(rf'\b{cmd}\b', command):
            return (
                f"Command blocked: '{cmd}' is not allowed in sandbox mode. "
                f"Only standard development commands are permitted."
            )

    return None


def convert_virtual_paths_in_command(command: str) -> str:
    """
    Convert virtual paths (starting with /) in commands to relative paths.

    Examples:
    - "python /main.py" -> "python ./main.py"
    - "cat /data/file.txt" -> "cat ./data/file.txt"
    - "ls /" -> "ls ."
    - "python main.py" -> "python main.py" (unchanged)

    Args:
        command: Original command

    Returns:
        Converted command
    """

    def replace_virtual_path(match):
        path = match.group(0)

        # Skip content that looks like a URL
        if '://' in command[max(0, match.start() - 10):match.end() + 10]:
            return path

        # Convert virtual path
        if path == '/':
            return '.'
        else:
            return '.' + path

    # Match pattern: paths starting with / (but not URLs)
    pattern = r'(?<=\s)/[^\s;|&<>\'"`]*|^/[^\s;|&<>\'"`]*'
    converted = re.sub(pattern, replace_virtual_path, command)

    return converted


class ReadOnlyFilesystemBackend(FilesystemBackend):
    """
    Read-only filesystem backend.

    Allows read, ls, grep, glob operations but blocks write and edit.
    Used for skills directory — agent can read skill definitions but cannot
    modify them.
    """

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(
            error="This directory is read-only. Write operations are not permitted here."
        )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return EditResult(
            error="This directory is read-only. Edit operations are not permitted here."
        )


class MergedReadOnlyBackend(BackendProtocol):
    """Read-only backend that merges two directories.

    Reads from *primary* first (user skills in workspace/skills/),
    falls back to *secondary* (system skills in ./skills/).
    User skills override system skills with the same name.

    Both directories share the same virtual path namespace — the agent
    sees all skills under /skills/ regardless of which backend serves them.
    """

    def __init__(self, primary_dir: str, secondary_dir: str):
        self._primary = ReadOnlyFilesystemBackend(root_dir=primary_dir, virtual_mode=True)
        self._secondary = ReadOnlyFilesystemBackend(root_dir=secondary_dir, virtual_mode=True)

    # -- read: try primary first, fall back to secondary --

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        try:
            result = self._primary.read(file_path, offset, limit)
            if not result.startswith("Error:"):
                return result
        except (ValueError, FileNotFoundError, OSError):
            pass
        return self._secondary.read(file_path, offset, limit)

    # -- ls_info: merge both, primary wins on name conflicts --

    def ls_info(self, path: str = "/") -> list:
        secondary_items = {item["path"]: item for item in self._secondary.ls_info(path)}
        primary_items = {item["path"]: item for item in self._primary.ls_info(path)}
        secondary_items.update(primary_items)  # primary overrides
        return sorted(secondary_items.values(), key=lambda x: x["path"])

    # -- grep_raw: search both, deduplicate --

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list:
        results = self._secondary.grep_raw(pattern, path, glob)
        try:
            results += self._primary.grep_raw(pattern, path, glob)
        except Exception:
            pass
        return results

    # -- glob_info: merge both --

    def glob_info(self, pattern: str, path: str = "/") -> list:
        secondary = {item["path"]: item for item in self._secondary.glob_info(pattern, path)}
        try:
            primary = {item["path"]: item for item in self._primary.glob_info(pattern, path)}
            secondary.update(primary)
        except Exception:
            pass
        return sorted(secondary.values(), key=lambda x: x["path"])

    # -- write / edit: blocked --

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(
            error="This directory is read-only. Write operations are not permitted here."
        )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return EditResult(
            error="This directory is read-only. Edit operations are not permitted here."
        )

    # -- download / upload --

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files, trying primary then secondary."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            resp = self._primary.download_files([path])[0]
            if resp.error is not None:
                resp = self._secondary.download_files([path])[0]
            responses.append(resp)
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [
            FileUploadResponse(path=path, error="permission_denied")
            for path, _ in files
        ]


class CustomSandboxBackend(LocalShellBackend):
    """
    Custom sandbox backend - inherits LocalShellBackend with added safety.

    Features:
    - Inherits all file operations (ls, read, write, edit, grep, glob)
    - Inherits shell command execution with output truncation and timeout
    - Adds command validation to prevent directory traversal and dangerous operations
    - Adds path sanitization to auto-correct common LLM path mistakes
    - Compatible with LangGraph checkpointer (no thread locks)
    """

    def __init__(
        self,
        root_dir: str = ".",
        *,
        virtual_mode: bool = True,
        timeout: int = 300,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
    ):
        """
        Initialize custom sandbox backend.

        Args:
            root_dir: File system root directory
            virtual_mode: Whether to enable virtual path mode
            timeout: Command execution timeout in seconds
            max_output_bytes: Max output size before truncation (default 100KB)
            env: Extra environment variables for subprocess
            inherit_env: Whether to inherit parent process env (default True)
        """
        super().__init__(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            env=env,
            inherit_env=inherit_env,
        )
        # Override parent's "local-" prefix with our own
        self._sandbox_id = f"evosci-{uuid.uuid4().hex[:8]}"
        # Ensure working directory exists
        os.makedirs(str(self.cwd), exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        """Resolve path with sanitization to prevent nested directories.

        Intercepts all file operations (read, write, edit, ls, grep, glob).
        Auto-corrects common LLM path mistakes instead of crashing:
          1. /workspace/file.py           → /file.py
          2. /Users/name/.../workspace/f  → /f  (strip up to workspace/)
          3. /Users/name/file.py          → /file.py (keep basename)
        """
        # Auto-strip /workspace/ prefix to prevent nesting
        if key.startswith("/workspace/"):
            key = key[len("/workspace"):]  # "/workspace/main.py" → "/main.py"
        elif key == "/workspace":
            key = "/"

        # Auto-correct system absolute paths
        for prefix in _SYSTEM_PATH_PREFIXES:
            if key.startswith(prefix):
                # Try to extract path after "workspace/" or "workspace" at end
                marker = "/workspace/"
                idx = key.find(marker)
                if idx != -1:
                    key = "/" + key[idx + len(marker):]
                elif key.endswith("/workspace"):
                    key = "/"
                else:
                    # Fall back to basename
                    key = "/" + Path(key).name
                break

        return super()._resolve_path(key)

    def execute(self, command: str) -> ExecuteResponse:
        """
        Execute shell command in sandbox environment.

        Commands are validated before execution to prevent:
        - Directory traversal (../)
        - Access to paths outside workspace
        - Dangerous system commands

        Then delegates to LocalShellBackend.execute() for actual execution.
        """
        # Validate command safety
        error = validate_command(command)
        if error:
            return ExecuteResponse(
                output=error,
                exit_code=1,
                truncated=False,
            )

        # Convert virtual paths to relative paths
        if self.virtual_mode:
            command = convert_virtual_paths_in_command(command=command)

        # Delegate to parent for subprocess execution
        return super().execute(command)
