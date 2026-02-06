"""MCP (Model Context Protocol) client integration.

Loads MCP server configurations from YAML, connects via langchain-mcp-adapters,
and routes the resulting LangChain tools to the appropriate agents.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# User-level config path
USER_CONFIG_DIR = Path.home() / ".config" / "evoscientist"
USER_MCP_CONFIG = USER_CONFIG_DIR / "mcp.yaml"

# Regex for ${VAR} env var interpolation
ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

# Supported transport protocols
VALID_TRANSPORTS = {"stdio", "http", "streamable_http", "sse", "websocket"}


def _interpolate_env(value: str) -> str:
    """Replace ``${VAR}`` patterns in *value* with environment variable values.

    Missing variables are replaced with an empty string and a warning is logged.
    """
    def _replace(match: re.Match) -> str:
        var = match.group(1)
        val = os.environ.get(var)
        if val is None:
            logger.warning("MCP config: env var $%s is not set", var)
            return ""
        return val

    return ENV_VAR_RE.sub(_replace, value)


def _interpolate_value(value: Any) -> Any:
    """Recursively interpolate env vars in strings, dicts, and lists."""
    if isinstance(value, str):
        return _interpolate_env(value)
    if isinstance(value, dict):
        return {k: _interpolate_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_value(v) for v in value]
    return value


def _load_user_config() -> dict[str, Any]:
    """Load the user-level MCP config, returning an empty dict if absent."""
    if USER_MCP_CONFIG.is_file():
        try:
            data = yaml.safe_load(USER_MCP_CONFIG.read_text()) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_user_config(config: dict[str, Any]) -> None:
    """Write *config* to the user-level MCP config file."""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    USER_MCP_CONFIG.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def add_mcp_server(
    name: str,
    transport: str,
    *,
    command: str | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    tools: list[str] | None = None,
    expose_to: list[str] | None = None,
) -> dict[str, Any]:
    """Add or replace an MCP server in the user config.

    Returns the server entry that was written.
    """
    if transport not in VALID_TRANSPORTS:
        raise ValueError(
            f"Unknown transport {transport!r}. "
            f"Must be one of: {', '.join(sorted(VALID_TRANSPORTS))}"
        )

    entry: dict[str, Any] = {"transport": transport}

    if transport == "stdio":
        if not command:
            raise ValueError("stdio transport requires a command")
        entry["command"] = command
        entry["args"] = args or []
        if env:
            entry["env"] = env
    else:
        if not url:
            raise ValueError(f"{transport} transport requires a url")
        entry["url"] = url
        if headers:
            entry["headers"] = headers

    if tools:
        entry["tools"] = tools
    if expose_to:
        entry["expose_to"] = expose_to

    user_cfg = _load_user_config()
    user_cfg[name] = entry
    _save_user_config(user_cfg)
    return entry


def edit_mcp_server(name: str, **fields: Any) -> dict[str, Any]:
    """Update fields on an existing MCP server entry.

    Only the provided *fields* are changed; everything else is preserved.
    Passing ``None`` for a field removes it.

    Returns the updated entry.

    Raises:
        KeyError: if *name* doesn't exist in the user config.
        ValueError: on invalid transport or missing required fields.
    """
    user_cfg = _load_user_config()
    if name not in user_cfg:
        raise KeyError(f"MCP server {name!r} not found in user config")

    entry = user_cfg[name]

    for key, value in fields.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value

    # Re-validate after edits
    transport = entry.get("transport", "")
    if transport and transport not in VALID_TRANSPORTS:
        raise ValueError(
            f"Unknown transport {transport!r}. "
            f"Must be one of: {', '.join(sorted(VALID_TRANSPORTS))}"
        )
    if transport == "stdio" and not entry.get("command"):
        raise ValueError("stdio transport requires a command")
    if transport in ("http", "streamable_http", "sse", "websocket") and not entry.get("url"):
        raise ValueError(f"{transport} transport requires a url")

    user_cfg[name] = entry
    _save_user_config(user_cfg)
    return entry


def remove_mcp_server(name: str) -> bool:
    """Remove an MCP server from the user config.

    Returns True if removed, False if it didn't exist.
    """
    user_cfg = _load_user_config()
    if name not in user_cfg:
        return False
    del user_cfg[name]
    _save_user_config(user_cfg)
    return True


def parse_mcp_add_args(tokens: list[str]) -> dict:
    """Parse CLI tokens for ``/mcp add`` into kwargs for :func:`add_mcp_server`.

    Syntax::

        <name> <transport> <command-or-url> [extra-args...]
            [--tools t1,t2] [--expose-to a1,a2] [--header Key:Value]...
            [--env KEY=VALUE]...

    For stdio: positional args after transport are command + args.
    For http/sse/websocket: first positional arg after transport is url.
    """
    if len(tokens) < 3:
        raise ValueError(
            "Usage: <name> <transport> <command-or-url> [args...]\n"
            "  Options: --tools t1,t2  --expose-to agent1,agent2  --header Key:Value  --env KEY=VALUE"
        )

    name = tokens[0]
    transport = tokens[1]

    positional: list[str] = []
    tools: list[str] | None = None
    expose_to: list[str] | None = None
    headers: dict[str, str] = {}
    env: dict[str, str] = {}

    i = 2
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--tools" and i + 1 < len(tokens):
            tools = [t.strip() for t in tokens[i + 1].split(",") if t.strip()]
            i += 2
        elif tok == "--expose-to" and i + 1 < len(tokens):
            expose_to = [a.strip() for a in tokens[i + 1].split(",") if a.strip()]
            i += 2
        elif tok == "--header" and i + 1 < len(tokens):
            kv = tokens[i + 1]
            if ":" in kv:
                k, v = kv.split(":", 1)
                headers[k.strip()] = v.strip()
            i += 2
        elif tok == "--env" and i + 1 < len(tokens):
            kv = tokens[i + 1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k.strip()] = v.strip()
            i += 2
        else:
            positional.append(tok)
            i += 1

    kwargs: dict = {"name": name, "transport": transport}

    if transport == "stdio":
        if not positional:
            raise ValueError("stdio transport requires a command after the transport name")
        kwargs["command"] = positional[0]
        kwargs["args"] = positional[1:]
        if env:
            kwargs["env"] = env
    else:
        if not positional:
            raise ValueError(f"{transport} transport requires a url after the transport name")
        kwargs["url"] = positional[0]
        if headers:
            kwargs["headers"] = headers

    if tools:
        kwargs["tools"] = tools
    if expose_to:
        kwargs["expose_to"] = expose_to

    return kwargs


def parse_mcp_edit_args(tokens: list[str]) -> tuple[str, dict]:
    """Parse CLI tokens for ``/mcp edit`` into (name, fields).

    Syntax::

        <name> [--transport T] [--command C] [--url U]
               [--tools t1,t2] [--tools none] [--expose-to a1,a2]
               [--header Key:Value]... [--env KEY=VALUE]...

    ``--tools none`` and ``--expose-to none`` clear those fields.
    """
    if not tokens:
        raise ValueError(
            "Usage: <name> [--transport T] [--command C] [--url U] "
            "[--tools t1,t2] [--expose-to a1,a2] [--header K:V] [--env K=V]"
        )

    name = tokens[0]
    fields: dict[str, Any] = {}
    headers: dict[str, str] = {}
    env: dict[str, str] = {}

    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--transport" and i + 1 < len(tokens):
            fields["transport"] = tokens[i + 1]
            i += 2
        elif tok == "--command" and i + 1 < len(tokens):
            fields["command"] = tokens[i + 1]
            i += 2
        elif tok == "--url" and i + 1 < len(tokens):
            fields["url"] = tokens[i + 1]
            i += 2
        elif tok == "--args" and i + 1 < len(tokens):
            fields["args"] = tokens[i + 1].split(",")
            i += 2
        elif tok == "--tools" and i + 1 < len(tokens):
            val = tokens[i + 1]
            fields["tools"] = None if val == "none" else [t.strip() for t in val.split(",") if t.strip()]
            i += 2
        elif tok == "--expose-to" and i + 1 < len(tokens):
            val = tokens[i + 1]
            fields["expose_to"] = None if val == "none" else [a.strip() for a in val.split(",") if a.strip()]
            i += 2
        elif tok == "--header" and i + 1 < len(tokens):
            kv = tokens[i + 1]
            if ":" in kv:
                k, v = kv.split(":", 1)
                headers[k.strip()] = v.strip()
            i += 2
        elif tok == "--env" and i + 1 < len(tokens):
            kv = tokens[i + 1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k.strip()] = v.strip()
            i += 2
        else:
            i += 1

    if headers:
        fields["headers"] = headers
    if env:
        fields["env"] = env

    if not fields:
        raise ValueError("No fields to edit. Use --transport, --command, --url, --tools, --expose-to, etc.")

    return name, fields


def load_mcp_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and merge MCP configuration.

    Merges package-level config (shipped with EvoScientist) with user-level
    config at ``~/.config/evoscientist/mcp.yaml``. User config wins on
    conflict.

    Returns an empty dict if no servers are configured (MCP is optional).
    """
    merged: dict[str, Any] = {}

    # 1. Package-level config
    if config_path:
        pkg_path = Path(config_path)
        if pkg_path.is_file():
            try:
                data = yaml.safe_load(pkg_path.read_text()) or {}
                if isinstance(data, dict):
                    merged.update(data)
            except Exception as exc:
                logger.warning("Failed to load MCP config %s: %s", pkg_path, exc)

    # 2. User-level config (overrides package-level)
    if USER_MCP_CONFIG.is_file():
        try:
            data = yaml.safe_load(USER_MCP_CONFIG.read_text()) or {}
            if isinstance(data, dict):
                merged.update(data)
        except Exception as exc:
            logger.warning("Failed to load user MCP config %s: %s", USER_MCP_CONFIG, exc)

    # Interpolate env vars across all values
    merged = _interpolate_value(merged)

    return merged


def _build_connections(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert our YAML config to ``MultiServerMCPClient`` connections format.

    Each server entry maps to one connection dict with the fields that
    ``MultiServerMCPClient`` expects for the given transport.

    Unknown transports are skipped with a warning.
    """
    connections: dict[str, dict[str, Any]] = {}

    for name, server in config.items():
        transport = server.get("transport", "")

        if transport == "stdio":
            conn: dict[str, Any] = {
                "transport": "stdio",
                "command": server.get("command", ""),
                "args": server.get("args", []),
            }
            if "env" in server:
                conn["env"] = server["env"]
            connections[name] = conn

        elif transport in ("http", "streamable_http"):
            conn = {
                "transport": transport,
                "url": server.get("url", ""),
            }
            if "headers" in server:
                conn["headers"] = server["headers"]
            connections[name] = conn

        elif transport == "sse":
            conn = {
                "transport": "sse",
                "url": server.get("url", ""),
            }
            if "headers" in server:
                conn["headers"] = server["headers"]
            connections[name] = conn

        elif transport == "websocket":
            conn = {
                "transport": "websocket",
                "url": server.get("url", ""),
            }
            connections[name] = conn

        else:
            logger.warning("MCP server %r: unknown transport %r, skipping", name, transport)

    return connections


def _filter_tools(tools: list, allowed_names: list[str] | None) -> list:
    """Filter tools by allowlist.

    If *allowed_names* is ``None``, all tools pass through.
    """
    if allowed_names is None:
        return tools
    allowed_set = set(allowed_names)
    return [t for t in tools if t.name in allowed_set]


def _route_tools(
    config: dict[str, Any],
    server_tools: dict[str, list],
) -> dict[str, list]:
    """Group filtered tools by target agent.

    Args:
        config: Full MCP config dict (server name -> server settings).
        server_tools: server name -> list of LangChain tools from that server.

    Returns:
        Dict mapping agent name -> list of tools. Key ``"main"`` targets the
        main EvoScientist agent; other keys match subagent names.
    """
    by_agent: dict[str, list] = {}

    for server_name, tools in server_tools.items():
        server_cfg = config.get(server_name, {})

        # Apply tool name filter
        allowed = server_cfg.get("tools")  # None means all
        filtered = _filter_tools(tools, allowed)

        # Determine target agents
        expose_to = server_cfg.get("expose_to", ["main"])
        if isinstance(expose_to, str):
            expose_to = [expose_to]

        for agent_name in expose_to:
            by_agent.setdefault(agent_name, []).extend(filtered)

    return by_agent


async def _load_tools(config: dict[str, Any]) -> dict[str, list]:
    """Connect to MCP servers and retrieve tools.

    Returns a dict of server name -> list of LangChain tools.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    connections = _build_connections(config)
    if not connections:
        return {}

    server_tools: dict[str, list] = {}
    client = MultiServerMCPClient(connections)  # type: ignore[invalid-argument-type]

    for server_name in connections:
        try:
            tools = await client.get_tools(server_name=server_name)
            server_tools[server_name] = tools
            logger.info(
                "MCP server %r: loaded %d tool(s)", server_name, len(tools)
            )
        except Exception as exc:
            logger.warning("MCP server %r: failed to load tools: %s", server_name, exc)
            server_tools[server_name] = []

    return server_tools


def load_mcp_tools(config_path: str | Path | None = None) -> dict[str, list]:
    """Load MCP tools and return them grouped by target agent.

    This is the main entry point. It:
    1. Loads and merges YAML configs (package + user level)
    2. Connects to each configured MCP server
    3. Filters tools per server allowlist
    4. Routes tools to target agents

    Returns:
        Dict mapping agent name -> list of LangChain ``BaseTool`` objects.
        Key ``"main"`` = main agent. Other keys = subagent names.
        Returns empty dict if no MCP servers are configured.
    """
    config = load_mcp_config(config_path)
    if not config:
        return {}

    # Run async loader — use nest_asyncio for Jupyter compatibility
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Inside an already-running event loop (e.g. Jupyter)
        import nest_asyncio
        nest_asyncio.apply()

    try:
        server_tools = asyncio.run(_load_tools(config))
    except Exception as exc:
        logger.warning("MCP tool loading failed: %s", exc)
        return {}

    return _route_tools(config, server_tools)
