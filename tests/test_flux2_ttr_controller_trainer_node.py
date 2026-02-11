import flux2_ttr_controller_trainer_node as trainer_node


class _FakeRuntime:
    def __init__(self, layer_specs, layer_ready, ready_flags):
        self.layer_specs = layer_specs
        self.layer_ready = layer_ready
        self._ready_flags = dict(ready_flags)

    def _refresh_layer_ready(self, layer_key: str) -> bool:
        return bool(self._ready_flags.get(layer_key, False))


def _extract_layer_idx(layer_key: str):
    if ":" not in layer_key:
        return None
    _, idx_text = layer_key.split(":", 1)
    try:
        return int(idx_text)
    except Exception:
        return None


def test_interpolate_gumbel_temperature_is_linear_between_endpoints():
    assert trainer_node.interpolate_gumbel_temperature(start=1.0, end=0.5, iteration=0, total_iterations=5) == 1.0
    assert trainer_node.interpolate_gumbel_temperature(start=1.0, end=0.5, iteration=4, total_iterations=5) == 0.5
    mid = trainer_node.interpolate_gumbel_temperature(start=1.0, end=0.5, iteration=2, total_iterations=5)
    assert mid == 0.75


def test_summarize_step_ttr_uses_sigma_buckets_and_fallback_sigma():
    summary = trainer_node.summarize_step_ttr([(1.0, 0.9), (0.4, 0.4), (0.1, 0.1)], sigma_fallback=0.33)
    assert summary["ttr_ratio_high_sigma"] == 0.9
    assert summary["ttr_ratio_low_sigma"] == 0.1
    assert summary["ttr_ratio_spread"] == 0.8
    assert summary["steps_per_trajectory"] == 3.0

    empty_summary = trainer_node.summarize_step_ttr([], sigma_fallback=0.33)
    assert empty_summary["sigma"] == 0.33
    assert empty_summary["steps_per_trajectory"] == 0.0


def test_build_layer_index_to_key_falls_back_to_layer_ready_when_specs_absent():
    runtime = _FakeRuntime(
        layer_specs=["double:0", "other"],
        layer_ready=["single:2", "single:5", "bad"],
        ready_flags={},
    )
    mapping = trainer_node.build_layer_index_to_key(runtime, extract_layer_idx=_extract_layer_idx)
    assert mapping == {2: "single:2", 5: "single:5"}


def test_build_ready_layer_indices_filters_by_runtime_refresh_state():
    runtime = _FakeRuntime(
        layer_specs=["single:0", "single:1", "single:2"],
        layer_ready=[],
        ready_flags={"single:0": True, "single:1": False, "single:2": True},
    )
    ready_indices = trainer_node.build_ready_layer_indices(
        runtime,
        {0: "single:0", 1: "single:1", 2: "single:2"},
    )
    assert ready_indices == [0, 2]
