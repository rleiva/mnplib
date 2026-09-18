import math

import pytest

from mnplib.mismodel import mismodel


@pytest.mark.parametrize('inaccuracy,surfeit', [(0, 0), (0.3, 0.4), (1, 1)])
def test_mismodel_is_root_mean_square(inaccuracy, surfeit):
    assert mismodel(inaccuracy=inaccuracy, surfeit=surfeit) == pytest.approx(
        math.sqrt((inaccuracy ** 2 + surfeit ** 2) / 2)
    )


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_components_propagate_nan(value):
    assert math.isnan(mismodel(inaccuracy=value, surfeit=0.1))
    assert math.isnan(mismodel(inaccuracy=0.1, surfeit=value))


def test_negative_components_are_invalid():
    with pytest.raises(ValueError, match='nonnegative'):
        mismodel(inaccuracy=-0.1, surfeit=0.1)
