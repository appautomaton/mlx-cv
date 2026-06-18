import pytest

from mlx_cv import MultiplexBucket, ObjectMultiplexState, TrackMemoryRecord


def test_multiplex_state_assigns_objects_to_fixed_capacity_buckets():
    state = ObjectMultiplexState.from_object_ids([10, 11, 12], bucket_capacity=2, frame_index=0)

    assert state.active_object_ids == (10, 11, 12)
    assert state.object_to_bucket == {10: 0, 11: 0, 12: 1}
    assert [bucket.object_ids for bucket in state.buckets] == [(10, 11), (12,)]
    assert state.to_dict()["buckets"][0]["capacity"] == 2


def test_multiplex_state_records_frame_memory_metadata():
    state = ObjectMultiplexState.from_object_ids([7], bucket_capacity=4)
    state.add_memory(TrackMemoryRecord(7, 3, mask_shape=(8, 9), score=0.5, metadata={"source": "fixture"}))

    out = state.to_dict()
    assert out["frame_index"] == 3
    assert out["memory"] == [
        {
            "object_id": 7,
            "frame_index": 3,
            "mask_shape": [8, 9],
            "score": 0.5,
            "metadata": {"source": "fixture"},
        }
    ]


def test_multiplex_state_validates_invalid_bucket_assignments():
    with pytest.raises(ValueError, match="object_to_bucket does not match buckets"):
        ObjectMultiplexState(
            bucket_capacity=2,
            buckets=[MultiplexBucket(0, 2, (1,))],
            object_to_bucket={1: 1},
        )
    with pytest.raises(ValueError, match="multiple multiplex buckets"):
        ObjectMultiplexState(
            bucket_capacity=2,
            buckets=[MultiplexBucket(0, 2, (1,)), MultiplexBucket(1, 2, (1,))],
        )
    with pytest.raises(ValueError, match="not assigned"):
        ObjectMultiplexState(bucket_capacity=2).add_memory(TrackMemoryRecord(99, 0))


def test_multiplex_bucket_validates_capacity_and_duplicates():
    with pytest.raises(ValueError, match="capacity"):
        MultiplexBucket(0, 1, (1, 2))
    with pytest.raises(ValueError, match="unique"):
        MultiplexBucket(0, 2, (1, 1))
