import pytest

import sweep_utils


def test_build_clocked_sweep_balanced():
    clock = [0.0] * 100
    values = [1.0, 2.0, 3.0]
    output = sweep_utils.build_clocked_sweep(clock, values)
    assert len(output) == 100
    assert output.count(1.0) == 34 or output.count(1.0) == 33
    assert output.count(2.0) == 33
    assert output.count(3.0) == 33 or output.count(3.0) == 34


def test_build_clocked_sweep_exact():
    clock = [0.0] * 6
    values = [1.0, 2.0, 3.0]
    output = sweep_utils.build_clocked_sweep(clock, values)
    assert output == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_parse_clock_and_values_from_string():
    clock, values = sweep_utils.parse_clock_and_values("0,1,2", "[0.1, 0.2]")
    assert clock == [0.0, 1.0, 2.0]
    assert values == [0.1, 0.2]


def test_parse_clock_integer_string():
    clock, values = sweep_utils.parse_clock_and_values("3", "0.1, 0.2")
    assert clock == [1.0, 2.0, 3.0]
    assert values == [0.1, 0.2]


def test_infer_clock_from_values():
    clock, values = sweep_utils.parse_clock_and_values("", "0.1, 0.2, 0.3")
    assert clock == [1.0, 2.0, 3.0]
    assert values == [0.1, 0.2, 0.3]


def test_parse_rejects_long_values():
    with pytest.raises(ValueError):
        sweep_utils.build_clocked_sweep([0.0], [1.0, 2.0])


def test_generate_seed_batch_is_deterministic_for_same_seed():
    a = sweep_utils.generate_seed_batch(1234, 5)
    b = sweep_utils.generate_seed_batch(1234, 5)
    assert a == b
    assert len(a) == 5


def test_generate_seed_batch_respects_bounds():
    out = sweep_utils.generate_seed_batch(42, 64, min_value=10, max_value=20)
    assert len(out) == 64
    assert all(10 <= x <= 20 for x in out)


def test_generate_seed_batch_validates_inputs():
    with pytest.raises(ValueError):
        sweep_utils.generate_seed_batch(0, 0)
    with pytest.raises(ValueError):
        sweep_utils.generate_seed_batch(0, 1, min_value=-1, max_value=10)
    with pytest.raises(ValueError):
        sweep_utils.generate_seed_batch(0, 1, min_value=10, max_value=9)
