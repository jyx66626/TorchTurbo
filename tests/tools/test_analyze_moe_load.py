"""CPU checks for actual and fixed-routing counterfactual loads."""

import numpy as np
import pytest

from tools.analyze_moe_load import aggregate_load, compare_phases, simulate_ep


def _metadata():
    ranks = list(range(128, 256))
    return {
        "ep_group_global_ranks": ranks,
        "world_size": 256,
        "sp_size": 128,
        "num_local_experts": 2,
        "expert_placement": [rank for rank in ranks for _ in range(2)],
        "hidden_size": 8192,
        "bytes_per_element": 2,
    }


def test_expert_and_rank_load_use_actual_placement():
    counts = np.array([[1, 2, 5, 4], [7, 11, 13, 3]], dtype=np.int32)
    expert, rank = aggregate_load(counts, [7, 3], [3, 7, 3, 7])
    np.testing.assert_array_equal(expert, [8, 13, 18, 7])
    np.testing.assert_array_equal(rank, [20, 26])


@pytest.mark.parametrize("ep_size", [64, 32])
@pytest.mark.parametrize("case,expected", [
    ("balanced", 0),
    ("all_zero", 0),
    ("distributed_hotspot", None),
    ("one_source", 0),
    ("neighbor_experts", 7),
])
def test_ep128_split_can_improve_preserve_or_worsen_rank_load(ep_size, case, expected):
    metadata = _metadata()
    counts = np.zeros((128, 256), dtype=np.int32)
    if case == "balanced":
        counts.fill(1)
    elif case == "distributed_hotspot":
        counts[:, 0] = 10
        expected = -10 * (128 - ep_size)
    elif case == "one_source":
        counts[0, 0] = 10
    elif case == "neighbor_experts":
        counts[0, 0] = 7
        counts[1, 2] = 10

    result = simulate_ep(counts, metadata, ep_size)
    assert result["kind"] == "counterfactual_fixed_routing"
    assert result["max_rank_load_delta"] == expected
    assert result["num_local_experts"] == 256 // ep_size
    subgroups = result["subgroups"]
    assert len(subgroups) == 128 // ep_size
    for offset, subgroup in enumerate(subgroups):
        start = offset * ep_size
        assert subgroup["global_ranks"] == metadata["ep_group_global_ranks"][start:start + ep_size]
        assert sum(subgroup["rank_load"]) == counts[start:start + ep_size].sum()
    stats = result["rank_stats"]
    if case == "balanced":
        assert stats["min"] == stats["max"] == stats["mean"] == 256
        assert stats["max_over_mean"] == 1
    elif case == "all_zero":
        assert stats["max_over_mean"] is None
    elif case == "distributed_hotspot":
        assert stats["max_over_mean"] == ep_size
        assert stats["hot_global_ranks"] == [group["global_ranks"][0] for group in subgroups]
        assert result["buffer_estimates"] == {
            "fc2_output_max_bytes": 10 * ep_size * 8192 * 2,
            "combine_permutation_max_bytes": 10 * ep_size * 8192 * 2,
            "combine_returned_packed_max_bytes": 10 * 8192 * 2,
        }


def test_ep_identity_and_invalid_mesh():
    counts = np.ones((128, 256), dtype=np.int32)
    metadata = _metadata()
    assert simulate_ep(counts, metadata, 128)["identity"]
    for ep_size in (0, 3, 256):
        with pytest.raises(ValueError, match="EP must divide"):
            simulate_ep(counts, metadata, ep_size)
    metadata["ep_group_global_ranks"].reverse()
    with pytest.raises(ValueError, match="contiguous world mesh"):
        simulate_ep(counts, metadata, 32)


def test_phase_comparison_detects_source_changes_even_if_loads_are_equal():
    forward = np.zeros((128, 256), dtype=np.int32)
    forward[0, 0] = 10
    recompute = forward.copy()
    recompute[0, 0] = 0
    recompute[1, 0] = 10
    result = compare_phases(forward, recompute, _metadata())
    assert result["num_changed_elements"] == 2
    assert result["max_abs_count_delta"] == 10
    assert result["max_abs_rank_load_delta"] == 0

    recompute[1, 0] = 3
    recompute[1, 2] = 7
    result = compare_phases(forward, recompute, _metadata())
    assert result["rank_load_delta"][:2] == [-7, 7]
    assert result["max_abs_rank_load_delta"] == 7
