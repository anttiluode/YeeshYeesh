import pytest

from yeesh.experiment import ExperimentConfig, run_experiment


@pytest.fixture(scope="module")
def receipt():
    return run_experiment(ExperimentConfig(random_repeats=16))


def test_receipt_contains_both_mechanistic_conditions(receipt):
    assert set(("wave", "diffusion")).issubset(receipt)
    assert len(receipt["wave"]["optimized_schedule"]) == len(receipt["config"]["gate_times"])
    assert len(receipt["diffusion"]["optimized_schedule"]) == len(receipt["config"]["gate_times"])


def test_canonical_wave_result_beats_free_fixed_and_random_median(receipt):
    held = receipt["wave"]["heldout"]
    assert held["optimized"]["target_energy"] > held["free"]["target_energy"]
    assert held["optimized"]["selectivity"] > held["free"]["selectivity"]
    assert held["optimized"]["target_energy"] > held["fixed"]["target_energy"]
    assert held["optimized"]["target_energy"] > held["random_location_median"]["target_energy"]


def test_wave_order_controls_destroy_most_of_the_gain(receipt):
    held = receipt["wave"]["heldout"]
    assert held["optimized"]["target_energy"] > held["reverse"]["target_energy"]
    assert held["optimized"]["target_energy"] > held["random_order_median"]["target_energy"]


def test_diffusion_control_does_not_create_target_energy_gain(receipt):
    held = receipt["diffusion"]["heldout"]
    assert held["optimized"]["target_energy"] <= held["free"]["target_energy"] + 1e-12
    assert held["optimized"]["selectivity"] > held["free"]["selectivity"]


def test_operator_diagnostics_record_real_operator_change(receipt):
    diag = receipt["wave"]["operator_diagnostics"]
    assert diag["frobenius_difference"] > 1e-6
    assert diag["controlled_spectral_norm"] <= 1.0 + 1e-10
    assert diag["free_spectral_norm"] == pytest.approx(1.0, abs=1e-10)
