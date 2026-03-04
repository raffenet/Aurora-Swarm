"""Hostfile parser for Aurora agent endpoints.

Hostfile format — one agent per line (tab-delimited)::

    host1\t8000\tnode=aurora-0001\trole=worker
    host2\t8000\tnode=aurora-0002\trole=critic

Blank lines and lines starting with ``#`` are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AgentEndpoint:
    """A single agent's network address plus optional metadata tags."""

    host: str
    port: int
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def parse_hostfile(path: str | Path) -> list[AgentEndpoint]:
    """Parse a hostfile and return a list of :class:`AgentEndpoint` objects.

    Parameters
    ----------
    path:
        Path to the hostfile.

    Returns
    -------
    list[AgentEndpoint]
        Parsed endpoints in file order.
    """
    endpoints: list[AgentEndpoint] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Support both tab-separated and whitespace-separated formats.
            # Also support "host:port" as the first token.
            parts = line.split('\t') if '\t' in line else line.split()
            first = parts[0]
            if ":" in first:
                # "host:port" combined token
                raw_host, _, raw_port = first.rpartition(":")
                host = raw_host
                port = int(raw_port) if raw_port.isdigit() else 8000
                tag_start = 1
            elif len(parts) > 1 and parts[1].isdigit():
                host = first
                port = int(parts[1])
                tag_start = 2
            else:
                host = first
                port = 8000
                tag_start = 1
            # Parse optional tags from remaining columns
            tags: dict[str, str] = {}
            for token in parts[tag_start:]:
                if "=" in token:
                    key, value = token.split("=", 1)
                    tags[key] = value
            endpoints.append(AgentEndpoint(host=host, port=port, tags=tags))
    return endpoints
