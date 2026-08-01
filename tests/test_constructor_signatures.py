import inspect

import pytest

from screamer import EwZscore, RollingMean, RollingZscore


def test_wrapped_functors_expose_generated_constructor_signatures():
    ew_parameters = inspect.signature(EwZscore).parameters
    assert list(ew_parameters) == ["com", "span", "halflife", "alpha"]
    assert "window_size" not in ew_parameters
    assert all(parameter.default is None for parameter in ew_parameters.values())

    rolling_parameters = inspect.signature(RollingMean).parameters
    assert list(rolling_parameters) == ["window_size", "start_policy"]
    assert rolling_parameters["window_size"].default == 20


def test_ew_and_rolling_parameter_names_remain_distinct():
    with pytest.raises(TypeError):
        EwZscore(window_size=60)
    with pytest.raises(ValueError, match="Exactly one"):
        EwZscore()

    assert len(EwZscore(span=60)([1.0, 2.0, 3.0])) == 3
    assert len(RollingZscore(window_size=3)([1.0, 2.0, 3.0])) == 3
