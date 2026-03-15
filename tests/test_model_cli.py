"""
Tests for the CLI entry points (main functions) of simulation models.
Uses mocking to avoid actual file I/O and plotting during tests.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from models.clock_model import main as clock_main
from models.ising_model import main as ising_main
from models.xy_model import main as xy_main


@pytest.fixture
def mock_plt():
    # Axes for 1x3 plots
    mock_axes_3 = [MagicMock(), MagicMock(), MagicMock()]
    # Axes for 2x2 plot
    mock_axes_2x2 = MagicMock()
    mock_axes_2x2.flatten.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

    with patch('matplotlib.pyplot.subplots') as mock_s, \
         patch('matplotlib.pyplot.savefig') as mock_save, \
         patch('matplotlib.pyplot.colorbar'), \
         patch('matplotlib.pyplot.tight_layout'):

        def side_effect(rows, cols, **kwargs):
            if rows == 1 and cols == 3:
                return MagicMock(), mock_axes_3
            elif rows == 2 and cols == 2:
                return MagicMock(), mock_axes_2x2
            return MagicMock(), MagicMock()

        mock_s.side_effect = side_effect
        yield mock_s, mock_save

@pytest.fixture
def mock_os():
    with patch('os.makedirs') as mock_mkdir:
        yield mock_mkdir

def test_ising_main(mock_plt, mock_os):
    """Verify Ising main() executes with default arguments."""
    with patch('sys.argv', ['ising_model.py', '--size', '4', '--steps', '5']):
        ising_main()

    mock_s, mock_save = mock_plt
    assert mock_s.called
    assert mock_save.called
    assert mock_os.called

def test_xy_main(mock_plt, mock_os):
    """Verify XY main() executes with default arguments."""
    with patch('sys.argv', ['xy_model.py', '--size', '4', '--steps', '5']):
        xy_main()

    mock_s, mock_save = mock_plt
    assert mock_s.called
    assert mock_save.called
    assert mock_os.called

def test_clock_main(mock_plt, mock_os):
    """Verify Clock main() executes with default arguments."""
    with patch('sys.argv', ['clock_model.py', '--size', '4', '--steps', '5']):
        clock_main()

    mock_s, mock_save = mock_plt
    assert mock_s.called
    assert mock_save.called
    assert mock_os.called
