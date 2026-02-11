from __future__ import annotations

import logging
import sys
import types

import flux2_comet_logging


def test_ensure_experiment_reuses_cached_experiment_and_logs_params_once(monkeypatch):
    start_calls = []
    set_name_calls = []
    params_calls = []

    class _FakeExperiment:
        def set_name(self, name):
            set_name_calls.append(str(name))

        def log_parameters(self, params):
            params_calls.append(dict(params))

        def log_metrics(self, metrics, step=None):
            del metrics, step
            return None

        def end(self):
            return None

    def _fake_start(api_key=None, project_name=None, workspace=None, experiment_key=None):
        start_calls.append((api_key, project_name, workspace, experiment_key))
        return _FakeExperiment()

    fake_comet = types.ModuleType("comet_ml")
    fake_comet.start = _fake_start
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)

    namespace = "test_shared_comet"
    flux2_comet_logging.clear_namespace_state(namespace)

    logger = logging.getLogger("test_flux2_comet_logging")
    exp_a, disabled_a = flux2_comet_logging.ensure_experiment(
        namespace=namespace,
        logger=logger,
        component_name="TestComet",
        enabled=True,
        disabled=False,
        existing_experiment=None,
        api_key="api-key",
        project_name="proj",
        workspace="ws",
        experiment_key="exp-key",
        display_name="display-name",
        persist_experiment=True,
        params={"alpha": 1.0},
    )
    exp_b, disabled_b = flux2_comet_logging.ensure_experiment(
        namespace=namespace,
        logger=logger,
        component_name="TestComet",
        enabled=True,
        disabled=False,
        existing_experiment=None,
        api_key="api-key",
        project_name="proj",
        workspace="ws",
        experiment_key="exp-key",
        display_name="display-name",
        persist_experiment=True,
        params={"alpha": 1.0},
    )

    assert disabled_a is False
    assert disabled_b is False
    assert exp_a is not None
    assert exp_b is exp_a
    assert start_calls == [("api-key", "proj", "ws", "exp-key")]
    assert set_name_calls == ["display-name"]
    assert params_calls == [{"alpha": 1.0}]


def test_release_experiment_ends_and_clears_cache_when_not_persistent():
    end_calls = []

    class _FakeExperiment:
        def end(self):
            end_calls.append(True)

    namespace = "test_release_comet"
    flux2_comet_logging.clear_namespace_state(namespace)
    exp = _FakeExperiment()
    flux2_comet_logging.experiments_for(namespace)["exp-key"] = exp
    flux2_comet_logging.logged_param_keys_for(namespace).add("exp-key")

    flux2_comet_logging.release_experiment(
        namespace=namespace,
        logger=logging.getLogger("test_flux2_comet_logging"),
        component_name="TestComet",
        experiment=exp,
        enabled=True,
        persist_experiment=False,
        experiment_key="exp-key",
    )

    assert end_calls == [True]
    assert "exp-key" not in flux2_comet_logging.experiments_for(namespace)
    assert "exp-key" not in flux2_comet_logging.logged_param_keys_for(namespace)


def test_flatten_params_and_build_prefixed_metrics():
    flattened: dict[str, object] = {}
    flux2_comet_logging.flatten_params(
        "training",
        {
            "optimizer": {"lr": 1e-4},
            "layers": [0, 1, 2],
        },
        flattened,
    )
    assert flattened == {
        "training.optimizer.lr": 1e-4,
        "training.layers": "0,1,2",
    }

    payload = flux2_comet_logging.build_prefixed_metrics(
        "flux2ttr_controller",
        {"loss": 1.2, "bad": "text", "nan_value": float("nan")},
    )
    assert payload == {"flux2ttr_controller/loss": 1.2}


def test_safe_log_metrics_returns_false_on_experiment_error():
    class _FakeExperiment:
        def log_metrics(self, metrics, step=None):
            del metrics, step
            raise RuntimeError("boom")

    ok = flux2_comet_logging.safe_log_metrics(
        experiment=_FakeExperiment(),
        payload={"m": 1.0},
        step=3,
        logger=logging.getLogger("test_flux2_comet_logging"),
        component_name="TestComet",
    )
    assert ok is False
