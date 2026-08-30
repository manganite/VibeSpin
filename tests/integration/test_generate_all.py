"""
Contract tests for the ``generate_all`` data-generation entry point.

The value of a single regeneration command is that it stays complete.  These
tests hold it to the pipeline table in SCRIPTS.md, which is the documented
list of what the notebooks read, so that a new data-producing script cannot be
added to one and forgotten in the other.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from scripts.generate_all import DATA_PRODUCTS, select_products

ROOT = Path(__file__).resolve().parents[2]

#: Rows of the SCRIPTS.md pipeline table look like
#: ``| `scripts/xy/bkt_transition.py` | `results/xy/bkt_transition.npz` | ... |``
_PIPELINE_ROW = re.compile(
    r'^\|\s*`(scripts/[\w/]+\.py)`\s*\|\s*(?:`(results/[\w/.]+\.npz)`|\*\(figure only\)\*)\s*\|'
)


def _documented_npz_products() -> dict[str, str]:
    """Map script path to NPZ output for every data-producing row of SCRIPTS.md."""
    documented: dict[str, str] = {}
    for line in (ROOT / 'SCRIPTS.md').read_text(encoding='utf-8').splitlines():
        match = _PIPELINE_ROW.match(line)
        if match and match.group(2):
            documented[match.group(1)] = match.group(2)
    return documented


def test_pipeline_table_is_parseable() -> None:
    """Guard the parser itself: the table must yield a plausible number of rows."""
    documented = _documented_npz_products()
    assert len(documented) >= 15, (
        f'Only {len(documented)} data rows parsed from the SCRIPTS.md pipeline table; '
        'the table format probably changed and this test no longer sees it.'
    )


def test_generate_all_covers_every_documented_dataset() -> None:
    """Every NPZ the pipeline table documents must be produced by generate_all."""
    documented = _documented_npz_products()
    covered = {
        f'scripts/{product.key}.py': f'results/{product.output}'
        for product in DATA_PRODUCTS
    }
    assert covered == documented, (
        'generate_all and the SCRIPTS.md pipeline table disagree.\n'
        f'Missing from generate_all: {sorted(set(documented) - set(covered))}\n'
        f'Not in the table: {sorted(set(covered) - set(documented))}\n'
        f'Differing outputs: '
        f'{sorted(k for k in set(covered) & set(documented) if covered[k] != documented[k])}'
    )


def test_every_product_module_is_runnable() -> None:
    """Each entry must name an importable module that exposes a main()."""
    for product in DATA_PRODUCTS:
        module = importlib.import_module(product.module)
        assert callable(getattr(module, 'main', None)), (
            f'{product.module} has no callable main(), so python -m cannot run it.'
        )


def test_product_keys_are_unique_and_match_their_module() -> None:
    """Selector keys must be unique and derivable from the module path."""
    keys = [product.key for product in DATA_PRODUCTS]
    assert len(keys) == len(set(keys))
    for product in DATA_PRODUCTS:
        assert product.module == 'scripts.' + product.key.replace('/', '.')


def test_select_products_filters_by_substring() -> None:
    """--only keeps matching keys and --skip removes them afterwards."""
    only_xy = select_products(products=DATA_PRODUCTS, only=['xy/'], skip=None)
    assert only_xy and all(p.key.startswith('xy/') for p in only_xy)

    without_kinetics = select_products(
        products=DATA_PRODUCTS, only=['xy/'], skip=['ordering_kinetics'],
    )
    assert len(without_kinetics) == len(only_xy) - 1
    assert all('ordering_kinetics' not in p.key for p in without_kinetics)

    assert select_products(products=DATA_PRODUCTS, only=None, skip=None) == list(DATA_PRODUCTS)


def test_select_products_rejects_a_pattern_that_matches_nothing() -> None:
    """A typo in --only or --skip must fail loudly rather than silently do nothing."""
    with pytest.raises(ValueError, match='matches no data product'):
        select_products(products=DATA_PRODUCTS, only=['heisenberg'], skip=None)
    with pytest.raises(ValueError, match='matches no data product'):
        select_products(products=DATA_PRODUCTS, only=None, skip=['heisenberg'])
