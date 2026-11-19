"""Sealed-period guard. Task 11 extends this module."""
from __future__ import annotations


class SealViolation(Exception):
    """Raised when a screening/validation step would touch the sealed period."""
