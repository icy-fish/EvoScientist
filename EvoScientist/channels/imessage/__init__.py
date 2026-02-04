"""iMessage channel implementation for EvoScientist.

Uses imsg CLI via JSON-RPC for real-time message streaming.

Requirements:
- macOS only
- imsg CLI: brew install steipete/tap/imsg
- Full Disk Access permission
- Messages.app logged into iCloud
"""

from .channel_rpc import IMessageChannelRpc as IMessageChannel
from .channel_rpc import IMessageConfig
from .probe import probe_imessage, ProbeResult
from .targets import (
    parse_target,
    normalize_handle,
    normalize_e164,
    IMessageTarget,
    IMessageService,
)

__all__ = [
    "IMessageChannel",
    "IMessageConfig",
    "probe_imessage",
    "ProbeResult",
    "parse_target",
    "normalize_handle",
    "normalize_e164",
    "IMessageTarget",
    "IMessageService",
]
