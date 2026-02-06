"""Tests for EvoScientist.mcp_client module."""

import textwrap
from types import SimpleNamespace

import pytest
import yaml

from EvoScientist.mcp_client import (
    _interpolate_env,
    _filter_tools,
    _route_tools,
    _build_connections,
    load_mcp_config,
    add_mcp_server,
    edit_mcp_server,
    remove_mcp_server,
    parse_mcp_add_args,
    parse_mcp_edit_args,
)


# ---- _interpolate_env ----


class TestInterpolateEnv:
    def test_substitutes_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret123")
        assert _interpolate_env("Bearer ${MY_KEY}") == "Bearer secret123"

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        assert _interpolate_env("${HOST}:${PORT}") == "localhost:8080"

    def test_missing_var_returns_empty(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        assert _interpolate_env("${NONEXISTENT_VAR_XYZ}") == ""

    def test_no_vars_unchanged(self):
        assert _interpolate_env("plain text") == "plain text"

    def test_empty_string(self):
        assert _interpolate_env("") == ""


# ---- load_mcp_config ----


@pytest.fixture()
def _no_user_mcp(monkeypatch, tmp_path):
    """Isolate load_mcp_config tests from the real user config."""
    monkeypatch.setattr(
        "EvoScientist.mcp_client.USER_MCP_CONFIG",
        tmp_path / "no_user_mcp.yaml",
    )


class TestLoadMcpConfig:
    def test_missing_file_returns_empty(self, tmp_path, _no_user_mcp):
        result = load_mcp_config(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_valid_file_parses(self, tmp_path, _no_user_mcp):
        cfg = tmp_path / "mcp.yaml"
        cfg.write_text(textwrap.dedent("""\
            my-server:
              transport: stdio
              command: echo
              args: ["hello"]
        """))
        result = load_mcp_config(cfg)
        assert "my-server" in result
        assert result["my-server"]["transport"] == "stdio"

    def test_empty_file_returns_empty(self, tmp_path, _no_user_mcp):
        cfg = tmp_path / "mcp.yaml"
        cfg.write_text("")
        result = load_mcp_config(cfg)
        assert result == {}

    def test_comments_only_returns_empty(self, tmp_path, _no_user_mcp):
        cfg = tmp_path / "mcp.yaml"
        cfg.write_text("# just a comment\n# another comment\n")
        result = load_mcp_config(cfg)
        assert result == {}

    def test_env_var_interpolation(self, tmp_path, _no_user_mcp, monkeypatch):
        monkeypatch.setenv("TEST_TOKEN", "tok_abc")
        cfg = tmp_path / "mcp.yaml"
        cfg.write_text(textwrap.dedent("""\
            my-server:
              transport: http
              url: "http://localhost:8080/mcp"
              headers:
                Authorization: "Bearer ${TEST_TOKEN}"
        """))
        result = load_mcp_config(cfg)
        assert result["my-server"]["headers"]["Authorization"] == "Bearer tok_abc"

    def test_none_config_path_returns_empty(self, _no_user_mcp):
        result = load_mcp_config(None)
        assert result == {}


# ---- _build_connections ----


class TestBuildConnections:
    def test_stdio_connection(self):
        config = {
            "fs": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "server"],
            }
        }
        conns = _build_connections(config)
        assert "fs" in conns
        assert conns["fs"]["transport"] == "stdio"
        assert conns["fs"]["command"] == "npx"
        assert conns["fs"]["args"] == ["-y", "server"]

    def test_stdio_with_env(self):
        config = {
            "fs": {
                "transport": "stdio",
                "command": "npx",
                "args": [],
                "env": {"FOO": "bar"},
            }
        }
        conns = _build_connections(config)
        assert conns["fs"]["env"] == {"FOO": "bar"}

    def test_http_connection(self):
        config = {
            "api": {
                "transport": "http",
                "url": "http://localhost:8080/mcp",
                "headers": {"Authorization": "Bearer xxx"},
            }
        }
        conns = _build_connections(config)
        assert conns["api"]["transport"] == "http"
        assert conns["api"]["url"] == "http://localhost:8080/mcp"
        assert conns["api"]["headers"]["Authorization"] == "Bearer xxx"

    def test_sse_connection(self):
        config = {
            "sse-srv": {
                "transport": "sse",
                "url": "http://localhost:9090/sse",
            }
        }
        conns = _build_connections(config)
        assert conns["sse-srv"]["transport"] == "sse"
        assert conns["sse-srv"]["url"] == "http://localhost:9090/sse"

    def test_websocket_connection(self):
        config = {
            "ws": {
                "transport": "websocket",
                "url": "ws://localhost:8765",
            }
        }
        conns = _build_connections(config)
        assert conns["ws"]["transport"] == "websocket"

    def test_unknown_transport_skipped(self):
        config = {
            "bad": {
                "transport": "carrier_pigeon",
                "url": "coo://rooftop",
            }
        }
        conns = _build_connections(config)
        assert conns == {}

    def test_mixed_transports(self):
        config = {
            "a": {"transport": "stdio", "command": "cmd", "args": []},
            "b": {"transport": "http", "url": "http://x"},
            "c": {"transport": "unknown"},
        }
        conns = _build_connections(config)
        assert set(conns.keys()) == {"a", "b"}


# ---- _filter_tools ----


def _make_tool(name: str):
    """Create a minimal mock tool with a .name attribute."""
    return SimpleNamespace(name=name)


class TestFilterTools:
    def test_none_allowlist_passes_all(self):
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        assert _filter_tools(tools, None) == tools

    def test_allowlist_filters(self):
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        result = _filter_tools(tools, ["a", "c"])
        assert [t.name for t in result] == ["a", "c"]

    def test_empty_allowlist_filters_all(self):
        tools = [_make_tool("a"), _make_tool("b")]
        assert _filter_tools(tools, []) == []

    def test_allowlist_with_nonexistent_name(self):
        tools = [_make_tool("a")]
        result = _filter_tools(tools, ["a", "nonexistent"])
        assert [t.name for t in result] == ["a"]

    def test_empty_tools_list(self):
        assert _filter_tools([], ["a"]) == []
        assert _filter_tools([], None) == []


# ---- _route_tools ----


class TestRouteTools:
    def test_default_routes_to_main(self):
        config = {"srv": {"transport": "stdio"}}
        server_tools = {"srv": [_make_tool("x")]}
        result = _route_tools(config, server_tools)
        assert "main" in result
        assert [t.name for t in result["main"]] == ["x"]

    def test_expose_to_named_agent(self):
        config = {"srv": {"transport": "stdio", "expose_to": ["code-agent"]}}
        server_tools = {"srv": [_make_tool("x"), _make_tool("y")]}
        result = _route_tools(config, server_tools)
        assert "code-agent" in result
        assert "main" not in result
        assert [t.name for t in result["code-agent"]] == ["x", "y"]

    def test_expose_to_multiple_agents(self):
        config = {"srv": {"transport": "stdio", "expose_to": ["main", "code-agent"]}}
        server_tools = {"srv": [_make_tool("x")]}
        result = _route_tools(config, server_tools)
        assert [t.name for t in result["main"]] == ["x"]
        assert [t.name for t in result["code-agent"]] == ["x"]

    def test_tool_filter_applied(self):
        config = {"srv": {"transport": "stdio", "tools": ["b"]}}
        server_tools = {"srv": [_make_tool("a"), _make_tool("b"), _make_tool("c")]}
        result = _route_tools(config, server_tools)
        assert [t.name for t in result["main"]] == ["b"]

    def test_multiple_servers(self):
        config = {
            "s1": {"transport": "stdio", "expose_to": ["main"]},
            "s2": {"transport": "http", "expose_to": ["research-agent"]},
        }
        server_tools = {
            "s1": [_make_tool("a")],
            "s2": [_make_tool("b")],
        }
        result = _route_tools(config, server_tools)
        assert [t.name for t in result["main"]] == ["a"]
        assert [t.name for t in result["research-agent"]] == ["b"]

    def test_expose_to_string_not_list(self):
        config = {"srv": {"transport": "stdio", "expose_to": "debug-agent"}}
        server_tools = {"srv": [_make_tool("x")]}
        result = _route_tools(config, server_tools)
        assert "debug-agent" in result

    def test_empty_server_tools(self):
        config = {"srv": {"transport": "stdio"}}
        server_tools = {"srv": []}
        result = _route_tools(config, server_tools)
        assert result.get("main", []) == []


# ---- add_mcp_server / remove_mcp_server ----


@pytest.fixture()
def user_mcp_dir(tmp_path, monkeypatch):
    """Redirect user MCP config to a temp directory."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "mcp.yaml"
    monkeypatch.setattr("EvoScientist.mcp_client.USER_CONFIG_DIR", cfg_dir)
    monkeypatch.setattr("EvoScientist.mcp_client.USER_MCP_CONFIG", cfg_file)
    return cfg_file


class TestAddMcpServer:
    def test_add_stdio_server(self, user_mcp_dir):
        entry = add_mcp_server(
            "fs", "stdio", command="npx", args=["-y", "server", "/tmp"]
        )
        assert entry["transport"] == "stdio"
        assert entry["command"] == "npx"
        assert entry["args"] == ["-y", "server", "/tmp"]
        # Verify persisted
        data = yaml.safe_load(user_mcp_dir.read_text())
        assert "fs" in data

    def test_add_http_server(self, user_mcp_dir):
        entry = add_mcp_server(
            "api", "http",
            url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer tok"},
        )
        assert entry["url"] == "http://localhost:8080/mcp"
        assert entry["headers"]["Authorization"] == "Bearer tok"

    def test_add_sse_server(self, user_mcp_dir):
        entry = add_mcp_server("sse-srv", "sse", url="http://localhost:9090/sse")
        assert entry["transport"] == "sse"

    def test_add_websocket_server(self, user_mcp_dir):
        entry = add_mcp_server("ws", "websocket", url="ws://localhost:8765")
        assert entry["transport"] == "websocket"

    def test_add_with_tools_and_expose_to(self, user_mcp_dir):
        entry = add_mcp_server(
            "fs", "stdio",
            command="npx", args=[],
            tools=["read_file"],
            expose_to=["main", "code-agent"],
        )
        assert entry["tools"] == ["read_file"]
        assert entry["expose_to"] == ["main", "code-agent"]

    def test_add_replaces_existing(self, user_mcp_dir):
        add_mcp_server("srv", "stdio", command="old")
        add_mcp_server("srv", "http", url="http://new")
        data = yaml.safe_load(user_mcp_dir.read_text())
        assert data["srv"]["transport"] == "http"

    def test_add_invalid_transport_raises(self, user_mcp_dir):
        with pytest.raises(ValueError, match="Unknown transport"):
            add_mcp_server("bad", "carrier_pigeon", url="coo://rooftop")

    def test_stdio_without_command_raises(self, user_mcp_dir):
        with pytest.raises(ValueError, match="requires a command"):
            add_mcp_server("bad", "stdio")

    def test_http_without_url_raises(self, user_mcp_dir):
        with pytest.raises(ValueError, match="requires a url"):
            add_mcp_server("bad", "http")

    def test_add_with_env(self, user_mcp_dir):
        entry = add_mcp_server(
            "fs", "stdio", command="npx", args=[], env={"FOO": "bar"}
        )
        assert entry["env"] == {"FOO": "bar"}

    def test_add_multiple_servers(self, user_mcp_dir):
        add_mcp_server("a", "stdio", command="cmd1")
        add_mcp_server("b", "http", url="http://x")
        data = yaml.safe_load(user_mcp_dir.read_text())
        assert "a" in data and "b" in data


class TestRemoveMcpServer:
    def test_remove_existing(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx")
        assert remove_mcp_server("fs") is True
        data = yaml.safe_load(user_mcp_dir.read_text()) or {}
        assert "fs" not in data

    def test_remove_nonexistent(self, user_mcp_dir):
        assert remove_mcp_server("nope") is False

    def test_remove_preserves_others(self, user_mcp_dir):
        add_mcp_server("a", "stdio", command="cmd1")
        add_mcp_server("b", "http", url="http://x")
        remove_mcp_server("a")
        data = yaml.safe_load(user_mcp_dir.read_text())
        assert "a" not in data
        assert "b" in data


# ---- _parse_mcp_add_args (CLI arg parser) ----


class TestParseMcpAddArgs:
    def test_stdio_basic(self):
        r = parse_mcp_add_args(["fs", "stdio", "npx", "-y", "server", "/tmp"])
        assert r["name"] == "fs"
        assert r["transport"] == "stdio"
        assert r["command"] == "npx"
        assert r["args"] == ["-y", "server", "/tmp"]

    def test_http_basic(self):
        r = parse_mcp_add_args(["api", "http", "http://localhost:8080/mcp"])
        assert r["url"] == "http://localhost:8080/mcp"

    def test_tools_flag(self):
        r = parse_mcp_add_args(["srv", "http", "http://x", "--tools", "a,b"])
        assert r["tools"] == ["a", "b"]

    def test_expose_to_flag(self):
        r = parse_mcp_add_args(["srv", "http", "http://x", "--expose-to", "main,code-agent"])
        assert r["expose_to"] == ["main", "code-agent"]

    def test_header_flag(self):
        r = parse_mcp_add_args(["srv", "http", "http://x", "--header", "Authorization:Bearer tok"])
        assert r["headers"] == {"Authorization": "Bearer tok"}

    def test_env_flag(self):
        r = parse_mcp_add_args(["srv", "stdio", "cmd", "--env", "FOO=bar"])
        assert r["env"] == {"FOO": "bar"}

    def test_too_few_tokens_raises(self):
        with pytest.raises(ValueError, match="Usage"):
            parse_mcp_add_args(["fs", "stdio"])

    def test_stdio_missing_command_raises(self):
        with pytest.raises(ValueError, match="requires a command"):
            parse_mcp_add_args(["fs", "stdio", "--tools", "a"])

    def test_http_missing_url_raises(self):
        with pytest.raises(ValueError, match="requires a url"):
            parse_mcp_add_args(["srv", "http", "--tools", "a"])


# ---- edit_mcp_server ----


class TestEditMcpServer:
    def test_edit_expose_to(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx", args=[])
        entry = edit_mcp_server("fs", expose_to=["main", "code-agent"])
        assert entry["expose_to"] == ["main", "code-agent"]
        assert entry["command"] == "npx"  # unchanged

    def test_edit_tools(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx", args=[])
        entry = edit_mcp_server("fs", tools=["read_file"])
        assert entry["tools"] == ["read_file"]

    def test_edit_clear_tools(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx", tools=["read_file"])
        entry = edit_mcp_server("fs", tools=None)
        assert "tools" not in entry

    def test_edit_url(self, user_mcp_dir):
        add_mcp_server("api", "http", url="http://old:8080/mcp")
        entry = edit_mcp_server("api", url="http://new:9090/mcp")
        assert entry["url"] == "http://new:9090/mcp"
        assert entry["transport"] == "http"  # unchanged

    def test_edit_nonexistent_raises(self, user_mcp_dir):
        with pytest.raises(KeyError, match="not found"):
            edit_mcp_server("nope", tools=["a"])

    def test_edit_invalid_transport_raises(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx")
        with pytest.raises(ValueError, match="Unknown transport"):
            edit_mcp_server("fs", transport="carrier_pigeon")

    def test_edit_removes_required_field_raises(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx")
        with pytest.raises(ValueError, match="requires a command"):
            edit_mcp_server("fs", command=None)

    def test_edit_preserves_unrelated_fields(self, user_mcp_dir):
        add_mcp_server(
            "fs", "stdio", command="npx", args=["-y", "srv"],
            tools=["a"], expose_to=["main"],
        )
        entry = edit_mcp_server("fs", expose_to=["code-agent"])
        assert entry["tools"] == ["a"]
        assert entry["args"] == ["-y", "srv"]
        assert entry["expose_to"] == ["code-agent"]


# ---- parse_mcp_edit_args ----


class TestParseMcpEditArgs:
    def test_basic_field(self):
        name, fields = parse_mcp_edit_args(["srv", "--url", "http://new"])
        assert name == "srv"
        assert fields["url"] == "http://new"

    def test_tools_none_clears(self):
        _, fields = parse_mcp_edit_args(["srv", "--tools", "none"])
        assert fields["tools"] is None

    def test_expose_to_csv(self):
        _, fields = parse_mcp_edit_args(["srv", "--expose-to", "main,code-agent"])
        assert fields["expose_to"] == ["main", "code-agent"]

    def test_multiple_fields(self):
        _, fields = parse_mcp_edit_args(["srv", "--url", "http://x", "--tools", "a,b"])
        assert fields["url"] == "http://x"
        assert fields["tools"] == ["a", "b"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Usage"):
            parse_mcp_edit_args([])

    def test_no_fields_raises(self):
        with pytest.raises(ValueError, match="No fields"):
            parse_mcp_edit_args(["srv"])
