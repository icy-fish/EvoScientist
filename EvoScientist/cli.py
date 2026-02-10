"""
EvoScientist Agent CLI

Command-line interface with streaming output for the EvoScientist research agent.

Features:
- Thinking panel (blue) - shows model reasoning
- Tool calls with status indicators (green/yellow/red dots)
- Tool results in tree format with folding
- Response panel (green) - shows final response
- Thread ID support for multi-turn conversations
- Interactive mode with prompt_toolkit
- Configuration management (onboard, config commands)
"""

import asyncio
import logging
import os
import re
import queue
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer  # type: ignore[import-untyped]
from prompt_toolkit import PromptSession  # type: ignore[import-untyped]
from prompt_toolkit.completion import Completer, Completion  # type: ignore[import-untyped]
from prompt_toolkit.history import FileHistory  # type: ignore[import-untyped]
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory  # type: ignore[import-untyped]
from prompt_toolkit.formatted_text import HTML  # type: ignore[import-untyped]
from prompt_toolkit.shortcuts import CompleteStyle  # type: ignore[import-untyped]
from prompt_toolkit.styles import Style as PtStyle  # type: ignore[import-untyped]
from rich.panel import Panel  # type: ignore[import-untyped]
from rich.text import Text  # type: ignore[import-untyped]
from rich.table import Table  # type: ignore[import-untyped]

# Backward-compat re-exports (tests import these from EvoScientist.cli)
from .stream.state import SubAgentState, StreamState, _parse_todo_items, _build_todo_stats  # noqa: F401
from .stream.display import console, _run_streaming
from .paths import ensure_dirs, new_run_dir, default_workspace_dir, RUNS_DIR


def _shorten_path(path: str) -> str:
    """Shorten absolute path to relative path from current directory."""
    if not path:
        return path
    try:
        cwd = os.getcwd()
        if path.startswith(cwd):
            # Remove cwd prefix, keep the relative part
            rel = path[len(cwd):].lstrip(os.sep)
            # Add current dir name for context
            return os.path.join(os.path.basename(cwd), rel) if rel else os.path.basename(cwd)
        return path
    except Exception:
        return path


# =============================================================================
# Background iMessage channel
# =============================================================================

_channel_logger = logging.getLogger(__name__)


@dataclass
class ChannelMessage:
    """Message from a channel (iMessage, Email, etc.)."""
    msg_id: str
    content: str
    sender: str
    channel_type: str  # "iMessage", "Email", "Slack"
    metadata: Any = None


class _ChannelState:
    """Singleton tracking background iMessage channel and message queue."""

    server = None       # IMessageServer | None
    thread = None       # threading.Thread | None
    loop = None         # asyncio.AbstractEventLoop | None
    agent = None        # shared agent reference (same as CLI)
    thread_id = None    # shared thread_id (same conversation as CLI)

    # Queue-based communication between channel thread and main CLI thread
    message_queue: queue.Queue = queue.Queue()
    pending_responses: dict = {}  # msg_id -> {"event": Event, "response": str | None}
    _response_lock = threading.Lock()

    @classmethod
    def is_running(cls) -> bool:
        return cls.thread is not None and cls.thread.is_alive()

    @classmethod
    def stop(cls):
        if cls.loop and cls.server:
            cls.loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(cls.server.stop())
            )
        if cls.thread:
            cls.thread.join(timeout=5)
        cls.server = None
        cls.thread = None
        cls.loop = None
        cls.agent = None
        cls.thread_id = None
        # Clear pending responses
        with cls._response_lock:
            for slot in cls.pending_responses.values():
                slot["event"].set()  # Unblock any waiting handlers
            cls.pending_responses.clear()

    @classmethod
    def enqueue(
        cls,
        content: str,
        sender: str,
        channel_type: str,
        metadata: Any = None,
    ) -> tuple[str, threading.Event]:
        """Enqueue a message from any channel for main thread processing.

        Returns:
            Tuple of (msg_id, event) - caller can wait on event for response.
        """
        msg_id = str(uuid.uuid4())
        event = threading.Event()
        with cls._response_lock:
            cls.pending_responses[msg_id] = {"event": event, "response": None}
        cls.message_queue.put(ChannelMessage(msg_id, content, sender, channel_type, metadata))
        return msg_id, event

    @classmethod
    def set_response(cls, msg_id: str, response: str) -> None:
        """Set response and signal completion."""
        with cls._response_lock:
            if msg_id in cls.pending_responses:
                cls.pending_responses[msg_id]["response"] = response
                cls.pending_responses[msg_id]["event"].set()

    @classmethod
    def get_response(cls, msg_id: str, timeout: float = 300) -> str | None:
        """Wait for and retrieve response.

        Args:
            msg_id: The message ID to get response for.
            timeout: Maximum seconds to wait (default 300 = 5 minutes).

        Returns:
            The response text, or None if timed out or not found.
        """
        with cls._response_lock:
            slot = cls.pending_responses.get(msg_id)
        if not slot:
            return None
        if slot["event"].wait(timeout=timeout):
            with cls._response_lock:
                return cls.pending_responses.pop(msg_id, {}).get("response")
        return None


def _run_channel_thread(server):
    """Entry point for background channel thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _ChannelState.loop = loop
    try:
        loop.run_until_complete(server.run())
    except Exception as e:
        _channel_logger.error(f"Channel error: {e}")
    finally:
        loop.close()


# =============================================================================
# Banner
# =============================================================================

EVOSCIENTIST_ASCII_LINES = [
    r" ███████╗ ██╗   ██╗  ██████╗  ███████╗  ██████╗ ██╗ ███████╗ ███╗   ██╗ ████████╗ ██╗ ███████╗ ████████╗",
    r" ██╔════╝ ██║   ██║ ██╔═══██╗ ██╔════╝ ██╔════╝ ██║ ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██║ ██╔════╝ ╚══██╔══╝",
    r" █████╗   ██║   ██║ ██║   ██║ ███████╗ ██║      ██║ █████╗   ██╔██╗ ██║    ██║    ██║ ███████╗    ██║   ",
    r" ██╔══╝   ╚██╗ ██╔╝ ██║   ██║ ╚════██║ ██║      ██║ ██╔══╝   ██║╚██╗██║    ██║    ██║ ╚════██║    ██║   ",
    r" ███████╗  ╚████╔╝  ╚██████╔╝ ███████║ ╚██████╗ ██║ ███████╗ ██║ ╚████║    ██║    ██║ ███████║    ██║   ",
    r" ╚══════╝   ╚═══╝    ╚═════╝  ╚══════╝  ╚═════╝ ╚═╝ ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚═╝ ╚══════╝    ╚═╝   ",
]

# Blue gradient: deep navy -> royal blue -> sky blue -> cyan
_GRADIENT_COLORS = ["#1a237e", "#1565c0", "#1e88e5", "#42a5f5", "#64b5f6", "#90caf9"]


def print_banner(
    thread_id: str,
    workspace_dir: str | None = None,
    memory_dir: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    provider: str | None = None,
):
    """Print welcome banner with ASCII art logo, thread ID, workspace path, and mode."""
    for line, color in zip(EVOSCIENTIST_ASCII_LINES, _GRADIENT_COLORS):
        console.print(Text(line, style=f"{color} bold"))
    info = Text()
    if model or provider or mode:
        info.append("  ", style="dim")
        parts = []
        if model:
            parts.append(("Model: ", model))
        if provider:
            parts.append(("Provider: ", provider))
        if mode:
            parts.append(("Mode: ", mode))
        for i, (label, value) in enumerate(parts):
            if i > 0:
                info.append("  ", style="dim")
            info.append(label, style="dim")
            info.append(value, style="magenta")
    info.append("\n  Type ", style="#ffe082")
    info.append("/", style="#ffe082 bold")
    info.append(" for commands", style="#ffe082")
    console.print(info)
    console.print()


# =============================================================================
# Skill management commands
# =============================================================================


def _cmd_list_skills() -> None:
    """List installed user skills."""
    from .tools.skills_manager import list_skills
    from .paths import USER_SKILLS_DIR

    skills = list_skills(include_system=False)

    if not skills:
        console.print("[dim]No user-installed skills.[/dim]")
        console.print("[dim]Install with:[/dim] /install-skill <path-or-url>")
        console.print(f"[dim]Skills directory:[/dim] [cyan]{_shorten_path(str(USER_SKILLS_DIR))}[/cyan]")
        console.print()
        return

    console.print(f"[bold]User-Installed Skills[/bold] ({len(skills)}):")
    for skill in skills:
        console.print(f"  [green]{skill.name}[/green] - {skill.description}")
    console.print(f"\n[dim]Location:[/dim] [cyan]{_shorten_path(str(USER_SKILLS_DIR))}[/cyan]")
    console.print()


def _cmd_install_skill(source: str) -> None:
    """Install a skill from local path or GitHub URL."""
    from .tools.skills_manager import install_skill

    if not source:
        console.print("[red]Usage:[/red] /install-skill <path-or-url>")
        console.print("[dim]Examples:[/dim]")
        console.print("  /install-skill ./my-skill")
        console.print("  /install-skill https://github.com/user/repo/tree/main/skill-name")
        console.print("  /install-skill user/repo@skill-name")
        console.print()
        return

    console.print(f"[dim]Installing skill from:[/dim] {source}")

    result = install_skill(source)

    if result["success"]:
        console.print(f"[green]Installed:[/green] {result['name']}")
        console.print(f"[dim]Description:[/dim] {result.get('description', '(none)')}")
        console.print(f"[dim]Path:[/dim] [cyan]{_shorten_path(result['path'])}[/cyan]")
        console.print()
        console.print("[dim]Reload with /new to apply.[/dim]")
    else:
        console.print(f"[red]Failed:[/red] {result['error']}")
    console.print()


def _cmd_uninstall_skill(name: str) -> None:
    """Uninstall a user-installed skill."""
    from .tools.skills_manager import uninstall_skill

    if not name:
        console.print("[red]Usage:[/red] /uninstall-skill <skill-name>")
        console.print("[dim]Use /skills to see installed skills.[/dim]")
        console.print()
        return

    result = uninstall_skill(name)

    if result["success"]:
        console.print(f"[green]Uninstalled:[/green] {name}")
        console.print("[dim]Reload with /new to apply.[/dim]")
    else:
        console.print(f"[red]Failed:[/red] {result['error']}")
    console.print()


def _create_channel_handler():
    """Create iMessage handler that enqueues messages for main thread processing.

    The handler enqueues messages to the shared queue and waits for the main
    CLI thread to process them with full Rich Live streaming. This ensures
    channel messages get the same display quality as direct CLI input.

    Returns:
        Async handler function: (msg) -> str
    """

    async def handler(msg) -> str:
        # Enqueue for main thread to process with full Live streaming
        msg_id, event = _ChannelState.enqueue(
            content=msg.content,
            sender=msg.sender,
            channel_type="iMessage",
            metadata=msg.metadata,
        )

        # Wait indefinitely for main thread to process and set response
        # (no timeout - let the agent work as long as needed)
        await asyncio.to_thread(event.wait)

        # Get the response
        with _ChannelState._response_lock:
            response = _ChannelState.pending_responses.pop(msg_id, {}).get("response", "")

        return response if response else "(empty response)"

    return handler


def _cmd_channel(args: str, agent: Any, thread_id: str) -> None:
    """Start iMessage channel in background thread using the shared agent.

    CLI and iMessage share the same agent + thread_id (same conversation).
    When an iMessage arrives, the main CLI thread processes it with full
    Rich Live streaming — same experience as direct CLI input.

    Usage: /channel [--allow SENDER]
    """
    from .channels.imessage import IMessageConfig
    from .channels.imessage.serve import IMessageServer

    if _ChannelState.is_running():
        console.print("[dim]iMessage channel already running[/dim]")
        console.print("[dim]Use[/dim] /channel stop [dim]to disconnect[/dim]\n")
        return

    parts = args.split() if args else []
    allowed = set()

    for i, p in enumerate(parts):
        if p == "--allow" and i + 1 < len(parts):
            allowed.add(parts[i + 1])

    config = IMessageConfig(
        allowed_senders=allowed if allowed else None,
    )

    # Store shared agent reference — no separate agent creation
    _ChannelState.agent = agent
    _ChannelState.thread_id = thread_id

    server = IMessageServer(
        config,
        handler=_create_channel_handler(),
        send_thinking=True,
    )

    _ChannelState.server = server
    _ChannelState.thread = threading.Thread(
        target=_run_channel_thread,
        args=(server,),
        daemon=True,
    )
    _ChannelState.thread.start()

    console.print("[green]iMessage channel running in background[/green]")
    if allowed:
        console.print(f"[dim]Allowed:[/dim] {allowed}")
    else:
        console.print("[dim]Allowed: all senders[/dim]")
    console.print("[dim]Use[/dim] /channel stop [dim]to disconnect[/dim]\n")


def _mcp_list_servers() -> None:
    """Print a table of configured MCP servers."""
    from .mcp import load_mcp_config
    from .mcp.client import USER_MCP_CONFIG

    config = load_mcp_config()

    if not config:
        console.print("[dim]No MCP servers configured.[/dim]")
        console.print("[dim]Add one with:[/dim] /mcp add <name> <transport> <command-or-url> [args...]")
        console.print()
        return

    table = Table(title="MCP Servers", show_header=True)
    table.add_column("Server", style="cyan")
    table.add_column("Transport", style="green")
    table.add_column("Tools", style="yellow")
    table.add_column("Expose To", style="magenta")

    for name, server in config.items():
        transport = server.get("transport", "?")
        tools = server.get("tools")
        tools_str = ", ".join(tools) if tools else "(all)"
        expose_to = server.get("expose_to", ["main"])
        if isinstance(expose_to, str):
            expose_to = [expose_to]
        expose_str = ", ".join(expose_to)
        table.add_row(name, transport, tools_str, expose_str)

    console.print(table)
    console.print(f"\n[dim]Config file: {USER_MCP_CONFIG}[/dim]")
    console.print()


def _mcp_add_server_from_kwargs(
    kwargs: dict[str, Any],
    *,
    show_reload_hint: bool = False,
) -> bool:
    """Add an MCP server from prepared kwargs."""
    from .mcp import add_mcp_server

    try:
        entry = add_mcp_server(**kwargs)
        console.print(f"[green]Added MCP server:[/green] [cyan]{kwargs['name']}[/cyan] ({entry['transport']})")
        if show_reload_hint:
            console.print("[dim]Reload with /new to apply.[/dim]")
        return True
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return False


def _mcp_edit_server_fields(
    name: str,
    fields: dict[str, Any],
    *,
    show_reload_hint: bool = False,
) -> bool:
    """Edit an MCP server from prepared field updates."""
    from .mcp import edit_mcp_server

    if not fields:
        console.print("[red]No fields to edit. Use --transport, --command, --url, --tools, --expose-to, etc.[/red]")
        return False

    try:
        edit_mcp_server(name, **fields)
        console.print(f"[green]Updated MCP server:[/green] [cyan]{name}[/cyan]")
        for k, v in fields.items():
            console.print(f"  [dim]{k}:[/dim] {v}")
        if show_reload_hint:
            console.print("[dim]Reload with /new to apply.[/dim]")
        return True
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return False
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return False


def _mcp_remove_server(name: str, *, show_reload_hint: bool = False) -> bool:
    """Remove an MCP server by name."""
    from .mcp import remove_mcp_server

    clean_name = name.strip()
    if not clean_name:
        console.print("[red]Usage:[/red] /mcp remove <name>")
        return False

    if remove_mcp_server(clean_name):
        console.print(f"[green]Removed MCP server:[/green] [cyan]{clean_name}[/cyan]")
        if show_reload_hint:
            console.print("[dim]Reload with /new to apply.[/dim]")
        return True

    console.print(f"[red]Server not found:[/red] {clean_name}")
    return False


def _render_mcp_server_config_table(name: str, server: dict[str, Any]) -> None:
    """Render one MCP server config table."""
    table = Table(
        title=f"MCP Server: {name}",
        show_header=True,
        title_style="bold cyan",
    )
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("transport", str(server.get("transport", "(not set)")))
    if server.get("command"):
        table.add_row("command", str(server["command"]))
    if server.get("args"):
        table.add_row("args", " ".join(str(a) for a in server["args"]))
    if server.get("url"):
        table.add_row("url", str(server["url"]))
    if server.get("headers"):
        for k, v in server["headers"].items():
            table.add_row(f"header: {k}", str(v))
    if server.get("env"):
        for k, v in server["env"].items():
            table.add_row(f"env: {k}", str(v))

    tools = server.get("tools")
    table.add_row("tools", ", ".join(tools) if tools else "[dim](all)[/dim]")
    expose_to = server.get("expose_to", ["main"])
    if isinstance(expose_to, str):
        expose_to = [expose_to]
    table.add_row("expose_to", ", ".join(expose_to))

    console.print(table)
    console.print()


def _show_mcp_config(name: str = "", *, show_blank_line: bool = True) -> str:
    """Show MCP config details.

    Returns:
        "ok" when rendered, "empty" when no config exists, "missing" when
        a specific server name is requested but not found.
    """
    from .mcp import load_mcp_config
    from .mcp.client import USER_MCP_CONFIG

    config = load_mcp_config()
    if not config:
        console.print("[dim]No MCP servers configured.[/dim]")
        if show_blank_line:
            console.print()
        return "empty"

    name = name.strip()
    if name and name not in config:
        console.print(f"[red]Server not found:[/red] {name}")
        if show_blank_line:
            console.print()
        return "missing"

    servers = {name: config[name]} if name else config
    for srv_name, srv in servers.items():
        _render_mcp_server_config_table(srv_name, srv)

    console.print(f"[dim]Config file: {USER_MCP_CONFIG}[/dim]")
    if show_blank_line:
        console.print()
    return "ok"


def _cmd_mcp_add(args_str: str) -> None:
    """Handle ``/mcp add ...``."""
    import shlex
    from .mcp import parse_mcp_add_args

    if not args_str.strip():
        console.print("[bold]Usage:[/bold] /mcp add <name> <command-or-url> [args...]")
        console.print()
        console.print("[dim]Transport is auto-detected: URLs → http, commands → stdio[/dim]")
        console.print()
        console.print("[bold]Examples:[/bold]")
        console.print("  /mcp add sequential-thinking npx -y @modelcontextprotocol/server-sequential-thinking")
        console.print("  /mcp add docs-langchain https://docs.langchain.com/mcp")
        console.print("  /mcp add my-sse http://localhost:9090/sse --transport sse --expose-to research-agent")
        console.print()
        console.print("[dim]Options:[/dim]")
        console.print("  --transport T          Transport type (default: auto-detect)")
        console.print("  --tools t1,t2          Tool allowlist")
        console.print("  --expose-to a1,a2      Target agents (default: main)")
        console.print("  --header Key:Value     HTTP header (repeatable)")
        console.print("  --env KEY=VALUE        Env var for stdio (repeatable)")
        console.print("  --env-ref KEY          Env var as runtime ${KEY} reference (repeatable)")
        console.print()
        return

    try:
        tokens = shlex.split(args_str)
        kwargs = parse_mcp_add_args(tokens)
        _mcp_add_server_from_kwargs(kwargs, show_reload_hint=True)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
    console.print()


def _cmd_mcp_edit(args_str: str) -> None:
    """Handle ``/mcp edit <name> --field value ...``."""
    import shlex
    from .mcp import parse_mcp_edit_args

    if not args_str.strip():
        console.print("[bold]Usage:[/bold] /mcp edit <name> --<field> <value> ...")
        console.print()
        console.print("[dim]Fields:[/dim] --transport, --command, --url, --args, --tools, --expose-to, --header, --env")
        console.print("[dim]Use[/dim] --tools none [dim]or[/dim] --expose-to none [dim]to clear a field.[/dim]")
        console.print()
        console.print("[bold]Examples:[/bold]")
        console.print("  /mcp edit filesystem --expose-to main,code-agent")
        console.print("  /mcp edit filesystem --tools read_file,write_file")
        console.print("  /mcp edit my-api --url http://new-host:8080/mcp")
        console.print("  /mcp edit my-api --tools none")
        console.print()
        return

    try:
        tokens = shlex.split(args_str)
        name, fields = parse_mcp_edit_args(tokens)
        _mcp_edit_server_fields(name, fields, show_reload_hint=True)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
    console.print()


def _cmd_mcp_remove(name: str) -> None:
    """Handle ``/mcp remove <name>``."""
    _mcp_remove_server(name, show_reload_hint=True)
    console.print()


def _cmd_mcp_config(name: str) -> None:
    """Handle ``/mcp config [name]``."""
    _show_mcp_config(name, show_blank_line=True)


def _cmd_mcp(args: str) -> None:
    """Dispatch ``/mcp`` subcommands."""
    args = args.strip()
    if not args:
        _mcp_list_servers()
        return

    parts = args.split(maxsplit=1)
    subcmd = parts[0].lower()
    subargs = parts[1] if len(parts) > 1 else ""

    if subcmd == "list":
        _mcp_list_servers()
    elif subcmd == "add":
        _cmd_mcp_add(subargs)
    elif subcmd == "edit":
        _cmd_mcp_edit(subargs)
    elif subcmd == "remove":
        _cmd_mcp_remove(subargs)
    elif subcmd == "config":
        _cmd_mcp_config(subargs)
    else:
        console.print("[bold]MCP commands:[/bold]")
        console.print("  /mcp              List configured servers")
        console.print("  /mcp list         List configured servers")
        console.print("  /mcp config       Show detailed server config")
        console.print("  /mcp add ...      Add a server")
        console.print("  /mcp edit ...     Edit an existing server")
        console.print("  /mcp remove ...   Remove a server")
        console.print()


def _cmd_channel_stop() -> None:
    """Stop background iMessage channel."""
    if not _ChannelState.is_running():
        console.print("[dim]No channel running[/dim]\n")
        return
    _ChannelState.stop()
    console.print("[dim]iMessage channel stopped[/dim]\n")


# =============================================================================
# CLI commands
# =============================================================================

def _print_channel_panel(channels: list[tuple[str, bool, str]]) -> None:
    """Print a summary panel for active channels.

    Args:
        channels: List of (name, ok, detail) tuples.
    """
    lines: list[Text] = []
    all_ok = True
    for name, ok, detail in channels:
        line = Text()
        if ok:
            line.append("● ", style="green")
            line.append(name, style="bold")
        else:
            line.append("✗ ", style="yellow")
            line.append(name, style="bold yellow")
            all_ok = False
        if detail:
            line.append(f"  {detail}", style="dim")
        lines.append(line)

    body = Text("\n").join(lines)
    border = "green" if all_ok else "yellow"
    console.print(Panel(body, title="[bold]Channels[/bold]", border_style=border, expand=False))
    console.print()


def _auto_start_channel(agent: Any, thread_id: str, allowed_senders_csv: str) -> None:
    """Start iMessage channel automatically from config.

    Args:
        agent: Compiled agent graph.
        thread_id: Current thread ID.
        allowed_senders_csv: Comma-separated allowed senders (empty = all).
    """
    try:
        from .channels.imessage import IMessageConfig
        from .channels.imessage.serve import IMessageServer

        allowed: set[str] | None = None
        if allowed_senders_csv.strip():
            allowed = {s.strip() for s in allowed_senders_csv.split(",") if s.strip()}

        config = IMessageConfig(allowed_senders=allowed if allowed else None)

        _ChannelState.agent = agent
        _ChannelState.thread_id = thread_id

        server = IMessageServer(config, handler=_create_channel_handler(), send_thinking=True)
        _ChannelState.server = server
        _ChannelState.thread = threading.Thread(
            target=_run_channel_thread,
            args=(server,),
            daemon=True,
        )
        _ChannelState.thread.start()

        detail = ", ".join(sorted(allowed)) if allowed else "all senders"
        _print_channel_panel([("iMessage", True, detail)])
    except Exception as e:
        _print_channel_panel([("iMessage", False, str(e))])


_SLASH_COMMANDS = [
    ("/thread", "Show thread ID, workspace & memory dir"),
    ("/new", "Start a new session"),
    ("/skills", "List installed skills"),
    ("/install-skill", "Add a skill from path or GitHub"),
    ("/uninstall-skill", "Remove an installed skill"),
    ("/mcp", "Manage MCP servers"),
    ("/channel", "Configure messaging channels"),
    ("/exit", "Quit EvoScientist"),
]

_COMPLETION_STYLE = PtStyle.from_dict({
    "completion-menu": "bg:default noreverse nounderline noitalic",
    "completion-menu.completion": "bg:default #888888 noreverse",
    "completion-menu.completion.current": "bg:default default bold noreverse",
    "completion-menu.meta.completion": "bg:default #888888 noreverse",
    "completion-menu.meta.completion.current": "bg:default default bold noreverse",
    "scrollbar.background": "bg:default",
    "scrollbar.button": "bg:default",
})


class SlashCommandCompleter(Completer):
    """Autocomplete for slash commands — triggers when input starts with '/'."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in _SLASH_COMMANDS:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=f"{cmd:<40}",
                    display_meta=desc,
                )


def cmd_interactive(
    agent: Any,
    show_thinking: bool = True,
    workspace_dir: str | None = None,
    workspace_fixed: bool = False,
    mode: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    imessage_enabled: bool = False,
    imessage_allowed_senders: str = "",
    run_name: str | None = None,
) -> None:
    """Interactive conversation mode with streaming output.

    Args:
        agent: Compiled agent graph
        show_thinking: Whether to display thinking panels
        workspace_dir: Per-session workspace directory path
        workspace_fixed: If True, /new keeps the same workspace directory
        mode: Workspace mode ('daemon' or 'run'), displayed in banner
        model: Model name to display in banner
        provider: LLM provider name to display in banner
        imessage_enabled: Whether to auto-start iMessage channel
        imessage_allowed_senders: Comma-separated allowed senders
        run_name: Optional run name for /new session deduplication
    """
    import nest_asyncio
    nest_asyncio.apply()

    thread_id = str(uuid.uuid4())
    from .EvoScientist import MEMORY_DIR
    memory_dir = MEMORY_DIR
    print_banner(thread_id, workspace_dir, memory_dir, mode, model, provider)

    history_file = str(os.path.expanduser("~/.EvoScientist_history"))
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        completer=SlashCommandCompleter(),
        complete_style=CompleteStyle.COLUMN,
        complete_while_typing=True,
        style=_COMPLETION_STYLE,
    )

    def _print_separator():
        """Print a horizontal separator line spanning the terminal width."""
        width = console.size.width
        console.print(Text("\u2500" * width, style="dim"))

    # Mutable state for async loop
    state = {
        "agent": agent,
        "thread_id": thread_id,
        "workspace_dir": workspace_dir,
        "running": True,
    }

    def _process_channel_message(msg: ChannelMessage) -> None:
        """Process a message from a channel with full Live streaming."""
        # Move past the current prompt line to avoid interference with prompt_toolkit
        # Then move back up and clear that line
        sys.stdout.write("\n\033[A\033[2K\r")
        sys.stdout.flush()
        # Display prompt with channel source on second line
        console.print(f"[bold blue]>[/bold blue] {msg.content}")
        console.print(Text.assemble(
            ("[", "dim"),
            (f"{msg.channel_type}: Received from ", "dim"),
            (msg.sender, "cyan"),
            ("]", "dim"),
        ))
        _print_separator()
        console.print()

        # Build channel callbacks for intermediate messages (thinking + todo + files)
        on_thinking = None
        on_todo = None
        on_file_write = None
        if _ChannelState.is_running() and _ChannelState.server and _ChannelState.loop:
            def _send_thinking(thinking_text: str) -> None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        _ChannelState.server.send_thinking_message(
                            msg.sender, thinking_text, msg.metadata,
                        ),
                        _ChannelState.loop,
                    )
                except Exception:
                    pass  # Non-critical — don't break main flow

            def _send_todo(todo_items: list) -> None:
                try:
                    lines = [f"\U0001f4cb {len(todo_items)} tasks ongoing"]  # 📋
                    for i, item in enumerate(todo_items, 1):
                        content = item.get("content", "")
                        lines.append(f"{i}. {content}")
                    lines.append("\U0001f680")  # 🚀
                    formatted = "\n".join(lines)
                    asyncio.run_coroutine_threadsafe(
                        _ChannelState.server.send_todo_message(
                            msg.sender, formatted, msg.metadata,
                        ),
                        _ChannelState.loop,
                    )
                except Exception:
                    pass  # Non-critical — don't break main flow

            def _send_file(real_path: str) -> None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        _ChannelState.server.channel.send_media(
                            recipient=msg.sender, file_path=real_path,
                            metadata=msg.metadata,
                        ),
                        _ChannelState.loop,
                    )
                except Exception:
                    pass  # Non-critical — don't break main flow

            on_thinking = _send_thinking
            on_todo = _send_todo
            on_file_write = _send_file

        try:
            # Use SAME _run_streaming as CLI input — full Live experience
            response_text = _run_streaming(
                state["agent"], msg.content, state["thread_id"], show_thinking,
                interactive=True, on_thinking=on_thinking, on_todo=on_todo,
                on_file_write=on_file_write,
            )

            # Set response for channel handler to retrieve
            _ChannelState.set_response(msg.msg_id, response_text or "")
            # Show replied indicator
            console.print(Text.assemble(
                ("[", "dim"),
                (f"{msg.channel_type}: Replied to ", "dim"),
                (msg.sender, "cyan"),
                ("]", "dim"),
            ))
        except Exception as e:
            console.print(f"[red]Channel processing error: {e}[/red]")
            _ChannelState.set_response(msg.msg_id, f"Error: {e}")

        _print_separator()

    async def _check_channel_queue():
        """Background task to check channel queue periodically."""
        while state["running"]:
            try:
                msg = _ChannelState.message_queue.get_nowait()
                _process_channel_message(msg)
            except queue.Empty:
                pass
            await asyncio.sleep(0.1)  # Check every 100ms

    async def _async_main_loop():
        """Async main loop with prompt_async and channel queue checking."""
        # Start background queue checker
        queue_task = asyncio.create_task(_check_channel_queue())

        # Auto-start iMessage channel if enabled in config
        if imessage_enabled and not _ChannelState.is_running():
            _auto_start_channel(state["agent"], state["thread_id"], imessage_allowed_senders)

        try:
            _print_separator()
            while state["running"]:
                try:
                    user_input = await session.prompt_async(
                        HTML('<ansiblue><b>❯</b></ansiblue> ')
                    )
                    user_input = user_input.strip()

                    if not user_input:
                        # Erase the empty prompt line so it looks like nothing happened
                        sys.stdout.write("\033[A\033[2K\r")
                        sys.stdout.flush()
                        continue

                    _print_separator()

                    # Special commands
                    if user_input.lower() in ("/exit", "/quit", "/q"):
                        console.print("[dim]Goodbye![/dim]")
                        state["running"] = False
                        break

                    if user_input.lower() == "/new":
                        # New session: new thread; workspace only changes if not fixed
                        if not workspace_fixed:
                            state["workspace_dir"] = _create_session_workspace(run_name)
                        console.print("[dim]Loading new session...[/dim]")
                        state["agent"] = _load_agent(workspace_dir=state["workspace_dir"])
                        state["thread_id"] = str(uuid.uuid4())
                        # Sync shared refs if channel is running
                        if _ChannelState.is_running():
                            _ChannelState.agent = state["agent"]
                            _ChannelState.thread_id = state["thread_id"]
                        console.print(f"[green]New session:[/green] [yellow]{state['thread_id']}[/yellow]")
                        if state["workspace_dir"]:
                            console.print(f"[dim]Workspace:[/dim] [cyan]{_shorten_path(state['workspace_dir'])}[/cyan]\n")
                        continue

                    if user_input.lower() == "/thread":
                        console.print(f"[dim]Thread:[/dim] [yellow]{state['thread_id']}[/yellow]")
                        if state["workspace_dir"]:
                            console.print(f"[dim]Workspace:[/dim] [cyan]{_shorten_path(state['workspace_dir'])}[/cyan]")
                        if memory_dir:
                            console.print(f"[dim]Memory dir:[/dim] [cyan]{_shorten_path(memory_dir)}[/cyan]")
                        console.print()
                        continue

                    if user_input.lower() == "/skills":
                        _cmd_list_skills()
                        continue

                    if user_input.lower().startswith("/install-skill"):
                        source = user_input[len("/install-skill"):].strip()
                        _cmd_install_skill(source)
                        continue

                    if user_input.lower().startswith("/uninstall-skill"):
                        name = user_input[len("/uninstall-skill"):].strip()
                        _cmd_uninstall_skill(name)
                        continue

                    if user_input.lower().startswith("/mcp"):
                        _cmd_mcp(user_input[4:])
                        continue

                    if user_input.lower().startswith("/channel"):
                        args = user_input[len("/channel"):].strip()
                        if args.lower() == "stop":
                            _cmd_channel_stop()
                        else:
                            _cmd_channel(args, state["agent"], state["thread_id"])
                        continue

                    # Stream agent response
                    console.print()
                    _run_streaming(state["agent"], user_input, state["thread_id"], show_thinking, interactive=True)
                    _print_separator()

                except KeyboardInterrupt:
                    console.print("\n[dim]Goodbye![/dim]")
                    state["running"] = False
                    break
                except EOFError:
                    # Handle Ctrl+D
                    console.print("\n[dim]Goodbye![/dim]")
                    state["running"] = False
                    break
                except Exception as e:
                    error_msg = str(e)
                    if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                        console.print("[red]Error: API key not configured.[/red]")
                        console.print("[dim]Run [bold]EvoSci onboard[/bold] to set up your API key.[/dim]")
                        state["running"] = False
                        break
                    else:
                        console.print(f"[red]Error: {e}[/red]")
        finally:
            queue_task.cancel()
            try:
                await queue_task
            except asyncio.CancelledError:
                pass

    # Run the async main loop
    try:
        asyncio.run(_async_main_loop())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


def cmd_run(agent: Any, prompt: str, thread_id: str | None = None, show_thinking: bool = True, workspace_dir: str | None = None) -> None:
    """Single-shot execution with streaming display.

    Args:
        agent: Compiled agent graph
        prompt: User prompt
        thread_id: Optional thread ID (generates new one if None)
        show_thinking: Whether to display thinking panels
        workspace_dir: Per-session workspace directory path
    """
    thread_id = thread_id or str(uuid.uuid4())

    width = console.size.width
    sep = Text("\u2500" * width, style="dim")
    console.print(sep)
    console.print(Text(f"> {prompt}"))
    console.print(sep)
    console.print(f"[dim]Thread: {thread_id}[/dim]")
    if workspace_dir:
        console.print(f"[dim]Workspace: {_shorten_path(workspace_dir)}[/dim]")
    console.print()

    try:
        _run_streaming(agent, prompt, thread_id, show_thinking, interactive=False)
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            console.print("[red]Error: API key not configured.[/red]")
            console.print("[dim]Run [bold]EvoSci onboard[/bold] to set up your API key.[/dim]")
            raise typer.Exit(1)
        else:
            console.print(f"[red]Error: {e}[/red]")
            raise


# =============================================================================
# Agent loading helpers
# =============================================================================

def _deduplicate_run_name(name: str, runs_dir: Path = RUNS_DIR) -> str:
    """Return *name* if available, otherwise *name_1*, *name_2*, etc."""
    if not (runs_dir / name).exists():
        return name
    i = 1
    while (runs_dir / f"{name}_{i}").exists():
        i += 1
    return f"{name}_{i}"


def _create_session_workspace(name: str | None = None) -> str:
    """Create a per-session workspace directory and return its path.

    Args:
        name: Optional human-friendly run name.  Duplicates are resolved
              by appending ``_1``, ``_2``, etc.  Falls back to a timestamp
              if *name* is None.
    """
    if name:
        session_id = _deduplicate_run_name(name)
    else:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = str(new_run_dir(session_id))
    os.makedirs(workspace_dir, exist_ok=True)
    return workspace_dir


def _load_agent(workspace_dir: str | None = None):
    """Load the CLI agent (with InMemorySaver checkpointer for multi-turn).

    Args:
        workspace_dir: Optional per-session workspace directory.
    """
    from .EvoScientist import create_cli_agent
    return create_cli_agent(workspace_dir=workspace_dir)


# =============================================================================
# Typer app
# =============================================================================

app = typer.Typer(no_args_is_help=False, add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})

# Config subcommand group
config_app = typer.Typer(help="Configuration management commands", invoke_without_command=True)
app.add_typer(config_app, name="config")

# MCP subcommand group
_MCP_HELP = """\
Configure and manage MCP servers.

Examples:
  # Add a local MCP server (stdio auto-detected):
  EvoSci mcp add local-server python -- /path/to/server.py

  # Add an npx-based server:
  EvoSci mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking

  # Add an HTTP server (http auto-detected from URL):
  EvoSci mcp add docs-langchain https://docs.langchain.com/mcp

  # Add a stdio server with env vars (hardcoded):
  EvoSci mcp add my-server node --env API_KEY=xxx -- server.js

  # Add a server with runtime env ref (resolved from .env at startup):
  EvoSci mcp add brave-search npx --env-ref BRAVE_API_KEY -- -y @modelcontextprotocol/server-brave-search

  # Expose to a specific sub-agent (e.g. research-agent):
  EvoSci mcp add brave-search npx --env-ref BRAVE_API_KEY -e research-agent -- -y @modelcontextprotocol/server-brave-search

  # Expose to multiple agents:
  EvoSci mcp add local-server python -e main,research-agent,code-agent -- /path/to/server.py

  # Explicit transport override:
  EvoSci mcp add my-sse https://example.com/sse --transport sse

Sub-agents (-e): planner-agent | research-agent | code-agent | debug-agent | data-analysis-agent | writing-agent
"""
mcp_app = typer.Typer(help=_MCP_HELP, invoke_without_command=True)
app.add_typer(mcp_app, name="mcp")


# =============================================================================
# Onboard command
# =============================================================================

@app.command()
def onboard(
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip API key validation during setup"
    ),
):
    """Interactive setup wizard for EvoScientist.

    Guides you through configuring API keys, model selection,
    workspace settings, and agent parameters.
    """
    from .onboard import run_onboard
    run_onboard(skip_validation=skip_validation)


# =============================================================================
# Config commands
# =============================================================================

@config_app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context):
    """Configuration management commands."""
    if ctx.invoked_subcommand is None:
        config_list()


@config_app.command("list")
def config_list():
    """List all configuration values."""
    from .config import list_config, get_config_path

    config_data = list_config()

    table = Table(title="EvoScientist Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    # Mask API keys
    def format_value(key: str, value: Any) -> str:
        if "api_key" in key and value:
            return "***" + str(value)[-4:] if len(str(value)) > 4 else "***"
        if value == "":
            return "[dim](not set)[/dim]"
        return str(value)

    for key, value in config_data.items():
        table.add_row(key, format_value(key, value))

    console.print(table)
    console.print(f"\n[dim]Config file: {get_config_path()}[/dim]")


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="Configuration key to get")):
    """Get a single configuration value."""
    from .config import get_config_value

    value = get_config_value(key)
    if value is None:
        console.print(f"[red]Unknown key: {key}[/red]")
        raise typer.Exit(1)

    # Mask API keys
    if "api_key" in key and value:
        display_value = "***" + str(value)[-4:] if len(str(value)) > 4 else "***"
    elif value == "":
        display_value = "(not set)"
    else:
        display_value = str(value)

    console.print(f"[cyan]{key}[/cyan]: {display_value}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key to set"),
    value: str = typer.Argument(..., help="New value"),
):
    """Set a single configuration value."""
    from .config import set_config_value

    if set_config_value(key, value):
        console.print(f"[green]Set {key}[/green]")
    else:
        console.print(f"[red]Invalid key: {key}[/red]")
        raise typer.Exit(1)


@config_app.command("reset")
def config_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Reset configuration to defaults."""
    from .config import reset_config, get_config_path

    config_path = get_config_path()

    if not config_path.exists():
        console.print("[yellow]No config file to reset.[/yellow]")
        return

    if not yes:
        confirm = typer.confirm("Reset configuration to defaults?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    reset_config()
    console.print("[green]Configuration reset to defaults.[/green]")


@config_app.command("path")
def config_path():
    """Show the configuration file path."""
    from .config import get_config_path

    path = get_config_path()
    exists = path.exists()
    status = "[green]exists[/green]" if exists else "[dim]not created yet[/dim]"
    console.print(f"{path} ({status})")


# =============================================================================
# MCP commands
# =============================================================================

@mcp_app.callback(invoke_without_command=True)
def mcp_callback(ctx: typer.Context):
    """MCP server management commands."""
    if ctx.invoked_subcommand is None:
        mcp_list()


@mcp_app.command("list")
def mcp_list():
    """List configured MCP servers."""
    _mcp_list_servers()


@mcp_app.command("config")
def mcp_config(
    name: Optional[str] = typer.Argument(None, help="Server name (omit to show all)"),
):
    """Show detailed configuration for MCP servers.

    \b
    Examples:
      evosci mcp config             # Show all servers in detail
      evosci mcp config filesystem  # Show one server
    """
    status = _show_mcp_config(name or "", show_blank_line=False)
    if status == "empty":
        console.print("[dim]Add one with:[/dim] EvoSci mcp add <name> <transport> <command-or-url> [args...]")
        return
    if status == "missing":
        raise typer.Exit(1)


@mcp_app.command("add")
def mcp_add(
    name: str = typer.Argument(..., help="Server name"),
    target: str = typer.Argument(..., help="Command (stdio) or URL (http/sse)"),
    args: Optional[list[str]] = typer.Argument(None, help="Extra args for stdio command"),
    transport: Optional[str] = typer.Option(None, "--transport", "-T", help="Transport type (default: auto-detect)"),
    tools: Optional[str] = typer.Option(None, "--tools", "-t", help="Comma-separated tool allowlist"),
    expose_to: Optional[str] = typer.Option(None, "--expose-to", "-e", help="Comma-separated target agents"),
    header: Optional[list[str]] = typer.Option(None, "--header", "-H", help="HTTP header as Key:Value (repeatable)"),
    env: Optional[list[str]] = typer.Option(None, "--env", help="Env var as KEY=VALUE for stdio (repeatable)"),
    env_ref: Optional[list[str]] = typer.Option(None, "--env-ref", help="Env var name as ${NAME} runtime ref (repeatable)"),
):
    """Add an MCP server to user config.

    \b
    Transport is auto-detected: URLs default to http, commands default to stdio.

    \b
    Examples:
      evosci mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking
      evosci mcp add docs-langchain https://docs.langchain.com/mcp
      evosci mcp add my-sse https://example.com/sse --transport sse -e research-agent
      evosci mcp add brave-search npx --env-ref BRAVE_API_KEY -- -y @modelcontextprotocol/server-brave-search
    """
    from .mcp.client import _infer_transport

    if transport is None:
        transport = _infer_transport(target)

    kwargs: dict = {
        "name": name,
        "transport": transport,
    }

    if transport == "stdio":
        kwargs["command"] = target
        kwargs["args"] = list(args) if args else []
        if env or env_ref:
            env_dict = {}
            for e in (env or []):
                if "=" in e:
                    k, v = e.split("=", 1)
                    env_dict[k.strip()] = v.strip()
            for ref in (env_ref or []):
                env_dict[ref] = "${" + ref + "}"
            if env_dict:
                kwargs["env"] = env_dict
    else:
        kwargs["url"] = target
        if header:
            hdr_dict = {}
            for h in header:
                if ":" in h:
                    k, v = h.split(":", 1)
                    hdr_dict[k.strip()] = v.strip()
            if hdr_dict:
                kwargs["headers"] = hdr_dict

    if tools:
        kwargs["tools"] = [t.strip() for t in tools.split(",") if t.strip()]
    if expose_to:
        kwargs["expose_to"] = [a.strip() for a in expose_to.split(",") if a.strip()]

    if not _mcp_add_server_from_kwargs(kwargs, show_reload_hint=False):
        raise typer.Exit(1)


@mcp_app.command("edit")
def mcp_edit(
    name: str = typer.Argument(..., help="Server name to edit"),
    transport: Optional[str] = typer.Option(None, "--transport", help="New transport type"),
    command: Optional[str] = typer.Option(None, "--command", help="New command (stdio)"),
    url: Optional[str] = typer.Option(None, "--url", help="New URL (http/sse/websocket)"),
    tools: Optional[str] = typer.Option(None, "--tools", "-t", help="Comma-separated tool allowlist ('none' to clear)"),
    expose_to: Optional[str] = typer.Option(None, "--expose-to", "-e", help="Comma-separated target agents ('none' to clear)"),
    header: Optional[list[str]] = typer.Option(None, "--header", "-H", help="HTTP header as Key:Value (repeatable)"),
    env: Optional[list[str]] = typer.Option(None, "--env", help="Env var as KEY=VALUE for stdio (repeatable)"),
):
    """Edit an existing MCP server in user config.

    \b
    Examples:
      evosci mcp edit filesystem --expose-to main,code-agent
      evosci mcp edit filesystem -t read_file,write_file
      evosci mcp edit my-api --url http://new-host:9090/mcp
      evosci mcp edit my-api --tools none
    """
    fields: dict = {}
    if transport is not None:
        fields["transport"] = transport
    if command is not None:
        fields["command"] = command
    if url is not None:
        fields["url"] = url
    if tools is not None:
        fields["tools"] = None if tools == "none" else [t.strip() for t in tools.split(",") if t.strip()]
    if expose_to is not None:
        fields["expose_to"] = None if expose_to == "none" else [a.strip() for a in expose_to.split(",") if a.strip()]
    if header:
        hdr_dict = {}
        for h in header:
            if ":" in h:
                k, v = h.split(":", 1)
                hdr_dict[k.strip()] = v.strip()
        if hdr_dict:
            fields["headers"] = hdr_dict
    if env:
        env_dict = {}
        for e in env:
            if "=" in e:
                k, v = e.split("=", 1)
                env_dict[k.strip()] = v.strip()
        if env_dict:
            fields["env"] = env_dict

    if not _mcp_edit_server_fields(name, fields, show_reload_hint=False):
        raise typer.Exit(1)


@mcp_app.command("remove")
def mcp_remove(
    name: str = typer.Argument(..., help="Server name to remove"),
):
    """Remove an MCP server from user config."""
    if not _mcp_remove_server(name, show_reload_hint=False):
        raise typer.Exit(1)


# =============================================================================
# Main callback (default behavior)
# =============================================================================

@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    mode: Optional[str] = typer.Option(
        None,
        "-m",
        "--mode",
        help="Workspace mode: 'daemon' (persistent, default) or 'run' (isolated per-session)",
    ),
    name: Optional[str] = typer.Option(
        None,
        "-n",
        "--name",
        help="Name for this run (used as directory name instead of timestamp; requires --mode run)",
    ),
    prompt: Optional[str] = typer.Option(None, "-p", "--prompt", help="Query to execute (single-shot mode)"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id", help="Thread ID for conversation persistence"),
    workdir: Optional[str] = typer.Option(None, "--workdir", help="Override workspace directory for this session"),
    use_cwd: bool = typer.Option(False, "--use-cwd", help="Use current working directory as workspace"),
    no_thinking: bool = typer.Option(False, "--no-thinking", help="Disable thinking display"),
):
    """EvoScientist Agent - AI-powered research & code execution CLI."""
    # If a subcommand was invoked, don't run the default behavior
    if ctx.invoked_subcommand is not None:
        return

    from dotenv import load_dotenv, find_dotenv  # type: ignore[import-untyped]
    # find_dotenv() traverses up the directory tree to locate .env
    load_dotenv(find_dotenv(), override=True)

    # Load and apply configuration
    from .config import get_effective_config, apply_config_to_env

    # Build CLI overrides dict
    cli_overrides = {}
    if mode:
        cli_overrides["default_mode"] = mode
    if workdir:
        cli_overrides["default_workdir"] = workdir
    if no_thinking:
        cli_overrides["show_thinking"] = False

    config = get_effective_config(cli_overrides)
    apply_config_to_env(config)

    show_thinking = config.show_thinking if not no_thinking else False

    # Validate mutually exclusive options
    if workdir and use_cwd:
        raise typer.BadParameter("Use either --workdir or --use-cwd, not both.")

    if mode and (workdir or use_cwd):
        raise typer.BadParameter("--mode cannot be combined with --workdir or --use-cwd")

    if mode and mode not in ("run", "daemon"):
        raise typer.BadParameter("--mode must be 'run' or 'daemon'")

    # --name only makes sense in run mode
    if name and not (mode == "run" or (not mode and not workdir and not use_cwd and config.default_mode == "run")):
        raise typer.BadParameter("--name can only be used with --mode run")

    # Sanitize run name: allow alphanumeric, hyphens, underscores
    if name:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise typer.BadParameter("--name may only contain letters, digits, hyphens, and underscores")

    ensure_dirs()

    # Resolve effective mode from config (CLI mode already applied via overrides)
    effective_mode: str | None = None  # None means explicit --workdir/--use-cwd was used

    # Resolve workspace directory for this session
    # Priority: --use-cwd > --workdir > --mode (explicit) > default_workdir > default_mode
    if use_cwd:
        workspace_dir = os.getcwd()
        workspace_fixed = True
    elif workdir:
        workspace_dir = os.path.abspath(os.path.expanduser(workdir))
        os.makedirs(workspace_dir, exist_ok=True)
        workspace_fixed = True
    elif mode:
        # Explicit --mode overrides default_workdir
        effective_mode = mode
        workspace_root = config.default_workdir or str(default_workspace_dir())
        workspace_root = os.path.abspath(os.path.expanduser(workspace_root))
        if effective_mode == "run":
            runs_dir = Path(workspace_root, "runs")
            session_id = _deduplicate_run_name(name, runs_dir) if name else datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_dir = os.path.join(runs_dir, session_id)
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = False
        else:  # daemon
            workspace_dir = workspace_root
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = True
    elif config.default_workdir:
        # Use configured default workdir with configured mode
        workspace_root = os.path.abspath(os.path.expanduser(config.default_workdir))
        effective_mode = config.default_mode
        if effective_mode == "run":
            runs_dir = Path(workspace_root, "runs")
            session_id = _deduplicate_run_name(name, runs_dir) if name else datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_dir = os.path.join(runs_dir, session_id)
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = False
        else:  # daemon
            workspace_dir = workspace_root
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = True
    else:
        effective_mode = config.default_mode
        if effective_mode == "run":
            workspace_dir = _create_session_workspace(name)
            workspace_fixed = False
        else:  # daemon mode (default)
            workspace_dir = str(default_workspace_dir())
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = True

    # Load agent with session workspace
    console.print("[dim]Loading agent...[/dim]")
    agent = _load_agent(workspace_dir=workspace_dir)

    if prompt:
        # Single-shot mode: execute query and exit
        cmd_run(agent, prompt, thread_id=thread_id, show_thinking=show_thinking, workspace_dir=workspace_dir)
    else:
        # Interactive mode (default)
        cmd_interactive(
            agent,
            show_thinking=show_thinking,
            workspace_dir=workspace_dir,
            workspace_fixed=workspace_fixed,
            mode=effective_mode,
            model=config.model,
            provider=config.provider,
            imessage_enabled=config.imessage_enabled,
            imessage_allowed_senders=config.imessage_allowed_senders,
            run_name=name,
        )


def _configure_logging():
    """Configure logging with warning symbols for better visibility."""
    from rich.logging import RichHandler

    class DimWarningHandler(RichHandler):
        """Custom handler that renders warnings in dim style."""

        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno == logging.WARNING:
                # Use Rich console to print dim warning
                msg = record.getMessage()
                console.print(f"[dim yellow]\u26a0\ufe0f  Warning:[/dim yellow] [dim]{msg}[/dim]")
            else:
                super().emit(record)

    # Configure root logger to use our handler for WARNING and above
    handler = DimWarningHandler(console=console, show_time=False, show_path=False, show_level=False)
    handler.setLevel(logging.WARNING)

    # Apply to root logger (catches all loggers including deepagents)
    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicate output
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.WARNING)


def main():
    """CLI entry point — delegates to the Typer app."""
    import warnings
    warnings.filterwarnings("ignore", message=".*not known to support tools.*")
    warnings.filterwarnings("ignore", message=".*type is unknown and inference may fail.*")
    _configure_logging()
    app()


if __name__ == "__main__":
    main()
