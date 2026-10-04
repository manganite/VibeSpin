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


def test_clock_main_defaults_to_discrete_model(mock_plt, mock_os):
    """Clock main() must build the discrete model unless --continuous is given."""
    import models.clock_model as clock_module

    with patch('sys.argv', ['clock_model.py', '--size', '4', '--steps', '3']), \
         patch.object(
             clock_module, 'DiscreteClockSimulation',
             wraps=clock_module.DiscreteClockSimulation,
         ) as discrete_cls, \
         patch.object(
             clock_module, 'ClockSimulation', wraps=clock_module.ClockSimulation,
         ) as continuous_cls:
        clock_main()
    assert discrete_cls.called
    assert not continuous_cls.called


def test_clock_main_discrete_wolff(mock_plt, mock_os):
    """The discrete model supports the Wolff update from the CLI."""
    with patch(
        'sys.argv', ['clock_model.py', '--size', '4', '--steps', '3', '--update', 'wolff'],
    ):
        clock_main()


def test_clock_main_continuous_wolff_rejected(mock_plt, mock_os):
    """Continuous model with the default A=1.0 must refuse the Wolff update."""
    with patch(
        'sys.argv',
        ['clock_model.py', '--size', '4', '--steps', '3', '--update', 'wolff', '--continuous'],
    ), pytest.raises(ValueError, match='requires A=0.0'):
        clock_main()


def test_clock_main_aniso_requires_continuous(mock_plt, mock_os):
    """--aniso without --continuous is a usage error."""
    with patch('sys.argv', ['clock_model.py', '--size', '4', '--aniso', '0.5']), \
         pytest.raises(SystemExit):
        clock_main()
