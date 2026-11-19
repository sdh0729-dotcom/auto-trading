"""Sealed-period guard and the sealed-sample registry.

The sealed period is opened at most once per rule and at most `max_per_pocket_market`
(3) times per pocket x market. Every opening is persisted (atomic temp + rename) in a JSON
registry with its time and rule_id. Sealed data is handed out only by
`SealRegistry.load_sealed`, which accepts a ticket only if this registry's file holds the
exact same ticket, including a random token that a hand-built `SealTicket` will not have.
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from autotrade.types import Panel

_VERSION = 1


class SealViolation(Exception):
    """Raised when a screening/validation step would touch the sealed period."""


class SealReuseError(SealViolation):
    """The same rule_id already opened the seal for this pocket x market."""


class SealLimitError(SealViolation):
    """This pocket x market already used its quota of sealed final candidates."""


@dataclass(frozen=True)
class SealTicket:
    pocket: str
    market: str
    rule_id: str
    opened_at: str                                   # ISO-8601 UTC
    token: str = field(default="", repr=False)       # set only by SealRegistry.open


class SealRegistry:
    def __init__(self, path: Path, max_per_pocket_market: int = 3):
        self.path = Path(path)
        self.max_per_pocket_market = max_per_pocket_market

    def _load(self) -> list[dict]:
        """Re-read on every call so separate instances on the same file stay consistent."""
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != _VERSION:
            raise SealViolation(f"unrecognised seal registry format: {self.path}")
        return list(data["tickets"])

    def _save(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": _VERSION, "tickets": rows}, f, ensure_ascii=False,
                          indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def open(self, pocket: str, market: str, rule_id: str) -> SealTicket:
        rows = self._load()
        same = [r for r in rows if r["pocket"] == pocket and r["market"] == market]
        if any(r["rule_id"] == rule_id for r in same):
            raise SealReuseError(f"seal already opened for {pocket}/{market} rule {rule_id}")
        if len(same) >= self.max_per_pocket_market:
            raise SealLimitError(f"{pocket}/{market} already opened the seal "
                                 f"{len(same)} times (max {self.max_per_pocket_market})")
        ticket = SealTicket(pocket, market, rule_id,
                            datetime.now(timezone.utc).isoformat(), secrets.token_hex(16))
        self._save(rows + [asdict(ticket)])
        return ticket

    def tickets(self, pocket: str, market: str) -> list[SealTicket]:
        return [SealTicket(**r) for r in self._load()
                if r["pocket"] == pocket and r["market"] == market]

    def load_sealed(self, panel: Panel, periods, ticket: SealTicket) -> Panel:
        """The only sanctioned way to get the sealed slice `panel.slice(*periods.sealed)`."""
        if not isinstance(ticket, SealTicket) or not ticket.token:
            raise SealViolation("sealed data requires a ticket from SealRegistry.open")
        if asdict(ticket) not in self._load():
            raise SealViolation(f"ticket for rule {ticket.rule_id} was not issued by "
                                f"registry {self.path}")
        return panel.slice(*periods.sealed)
