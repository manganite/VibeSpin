"""Shared CLI parsing helpers for script entry points."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import cast


def parse_args_compat(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse CLI args with compatibility for parser wrappers used by some runners."""
    parse_arguments = getattr(parser, 'parse_arguments', None)
    if callable(parse_arguments):
        return cast(Callable[[], argparse.Namespace], parse_arguments)()
    return parser.parse_args()
