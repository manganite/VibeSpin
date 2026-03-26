"""Unit tests for CLI parsing helper utilities."""
from __future__ import annotations

import argparse
import sys

from utils.system import parse_args_compat


class _WrappedParser(argparse.ArgumentParser):
    """Parser wrapper exposing parse_arguments for compatibility checks."""

    def parse_arguments(self) -> argparse.Namespace:
        """Return a sentinel namespace to emulate external parser wrappers."""
        return argparse.Namespace(mode='wrapped')


def test_parse_args_compat_prefers_parse_arguments() -> None:
    """Helper should use parse_arguments when the parser wrapper provides it."""
    parser = _WrappedParser(prog='vibespin-test')

    args = parse_args_compat(parser)

    assert args.mode == 'wrapped'


def test_parse_args_compat_falls_back_to_parse_args(monkeypatch) -> None:
    """Helper should fall back to argparse parse_args for standard parsers."""
    parser = argparse.ArgumentParser(prog='vibespin-test')
    parser.add_argument('--size', type=int, default=16)
    monkeypatch.setattr(sys, 'argv', ['vibespin-test', '--size', '32'])

    args = parse_args_compat(parser)

    assert args.size == 32
