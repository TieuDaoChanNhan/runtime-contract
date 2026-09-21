"""Task sharding: several pods can serve one cell without splitting it unevenly or twice."""

import pytest

from codeact_runtime.benchmark.benchmark import _parse_shard


def test_shard_spec_parsing():
    assert _parse_shard(None) is None
    assert _parse_shard("0/3") == (0, 3)
    assert _parse_shard("2/3") == (2, 3)


@pytest.mark.parametrize("spec", ["3/3", "-1/3", "1/0", "x", "1", "1/2/3"])
def test_bad_shard_specs_are_rejected(spec):
    with pytest.raises(SystemExit):
        _parse_shard(spec)


@pytest.mark.parametrize("n_tasks,n_shards", [(25, 3), (25, 4), (16, 2), (7, 7), (5, 8)])
def test_shards_partition_the_task_list(n_tasks, n_shards):
    tasks = list(range(n_tasks))
    shards = [tasks[i::n_shards] for i in range(n_shards)]
    seen = [t for s in shards for t in s]
    assert sorted(seen) == tasks  # every task exactly once
    assert max(map(len, shards)) - min(map(len, shards)) <= 1  # and evenly
