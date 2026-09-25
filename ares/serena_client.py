"""Singleton stdio MCP client for Serena.

Keeps a single Serena subprocess alive for the lifetime of Ares.
All tool calls are serialised through a threading lock — safe for
the multi-agent environment where multiple agents might try to query
Serena concurrently.
"""

from __future__ import annotations

import json
import os
import logging
import subprocess
import threading
from typing import Any

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2024-11-05"


class SerenaClient:
    """Long-lived Serena MCP subprocess (stdio transport)."""

    def __init__(self, project: str | None = None) -> None:
        self._project = project
        self._lock = threading.Lock()
        self._next_id = 1
        self._proc: subprocess.Popen | None = None
        self._start()

    def _build_cmd(self) -> list[str]:
        cmd = [
            "serena", "start-mcp-server",
            "--transport", "stdio",
            "--enable-web-dashboard", "false",
            "--enable-gui-log-window", "false",
            "--log-level", "ERROR",
        ]
        if self._project:
            cmd += ["--project", self._project]
        return cmd

    def _start(self) -> None:
        logger.info("Starting Serena MCP subprocess...")
        self._proc = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._initialize()
        logger.info("Serena MCP ready.")

    def _send(self, msg: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def _recv_for_id(self, msg_id: int) -> dict:
        """Read lines until we find the JSON-RPC response matching msg_id."""
        assert self._proc and self._proc.stdout
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("Serena MCP process exited unexpectedly")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip any stray non-JSON lines (log output etc.)
            if msg.get("id") == msg_id:
                return msg

    def _initialize(self) -> None:
        msg_id = self._next_id
        self._next_id += 1
        self._send({
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "initialize",
            "params": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "ares", "version": "0.1.0"},
            },
        })
        self._recv_for_id(msg_id)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a Serena MCP tool and return its text output."""
        with self._lock:
            msg_id = self._next_id
            self._next_id += 1
            self._send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            })
            resp = self._recv_for_id(msg_id)

        if "error" in resp:
            err = resp["error"]
            raise RuntimeError(f"Serena '{name}' error: {err.get('message', err)}")

        contents = resp.get("result", {}).get("content", [])
        return "\n".join(c.get("text", "") for c in contents if c.get("type") == "text") or "(no output)"

    def close(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None


# ── Singleton ──────────────────────────────────────────────────────────────

_client: SerenaClient | None = None
_client_lock = threading.Lock()


def get_serena_client() -> SerenaClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                # Without a project Serena starts but every symbol tool fails.
                project = os.environ.get("ARES_SERENA_PROJECT") or os.getcwd()
                _client = SerenaClient(project=project)
    return _client
