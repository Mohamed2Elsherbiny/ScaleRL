"""Tests for the YAML experiment-config loader."""

import pytest

from scalerl.experiment import config_to_dict, load_experiment


def _write(tmp_path, text):
    p = tmp_path / "exp.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_defaults_when_minimal(tmp_path):
    path = _write(tmp_path, "algo: qlearning\n")
    exp = load_experiment(path)
    assert exp.algo == "qlearning"
    assert exp.train.algo == "qlearning"
    assert exp.env.max_steps == 500  # default preserved


def test_top_level_and_nested_overrides(tmp_path):
    path = _write(
        tmp_path,
        """
name: my-dqn
algo: dqn
seed: 7
total_steps: 12345
env:
  max_steps: 250
dqn:
  hidden_sizes: [128, 128]
  learning_rate: 0.0005
""",
    )
    exp = load_experiment(path)
    assert exp.name == "my-dqn"
    assert exp.seed == 7
    assert exp.train.total_steps == 12345
    assert exp.train.seed == 7
    assert exp.env.max_steps == 250
    # YAML list becomes a tuple for the tuple-typed field.
    assert exp.dqn.hidden_sizes == (128, 128)
    assert exp.dqn.learning_rate == 0.0005


def test_unknown_key_raises(tmp_path):
    path = _write(tmp_path, "env:\n  not_a_field: 1\n")
    with pytest.raises(ValueError):
        load_experiment(path)


def test_config_to_dict_is_json_friendly(tmp_path):
    path = _write(tmp_path, "algo: dqn\ndqn:\n  hidden_sizes: [32, 32]\n")
    exp = load_experiment(path)
    d = config_to_dict(exp.dqn)
    assert isinstance(d, dict)
    assert d["hidden_sizes"] == [32, 32]  # tuple -> list for JSON
