"""Typed temporal state helpers for video tracking models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TrackMemoryRecord",
    "MultiplexBucket",
    "ObjectMultiplexState",
]


@dataclass
class TrackMemoryRecord:
    """One per-object memory update at one video frame."""

    object_id: int
    frame_index: int
    mask_shape: tuple[int, int] | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.object_id = int(self.object_id)
        self.frame_index = int(self.frame_index)
        if self.object_id < 0:
            raise ValueError("TrackMemoryRecord.object_id must be non-negative")
        if self.frame_index < 0:
            raise ValueError("TrackMemoryRecord.frame_index must be non-negative")
        if self.mask_shape is not None:
            if len(self.mask_shape) != 2 or min(int(v) for v in self.mask_shape) < 1:
                raise ValueError("TrackMemoryRecord.mask_shape must be a positive (H,W) tuple")
            self.mask_shape = (int(self.mask_shape[0]), int(self.mask_shape[1]))
        if self.score is not None:
            self.score = float(self.score)


@dataclass
class MultiplexBucket:
    """Fixed-capacity Object Multiplex bucket."""

    index: int
    capacity: int
    object_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.index = int(self.index)
        self.capacity = int(self.capacity)
        self.object_ids = tuple(int(object_id) for object_id in self.object_ids)
        if self.index < 0:
            raise ValueError("MultiplexBucket.index must be non-negative")
        if self.capacity < 1:
            raise ValueError("MultiplexBucket.capacity must be positive")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("MultiplexBucket.object_ids must be unique")
        if any(object_id < 0 for object_id in self.object_ids):
            raise ValueError("MultiplexBucket.object_ids must be non-negative")
        if len(self.object_ids) > self.capacity:
            raise ValueError("MultiplexBucket.object_ids exceeds bucket capacity")

    @property
    def has_room(self) -> bool:
        return len(self.object_ids) < self.capacity

    def add(self, object_id: int) -> "MultiplexBucket":
        object_id = int(object_id)
        if object_id in self.object_ids:
            return self
        if not self.has_room:
            raise ValueError(f"MultiplexBucket {self.index} is full")
        return MultiplexBucket(self.index, self.capacity, self.object_ids + (object_id,))

    def remove(self, object_id: int) -> "MultiplexBucket":
        object_id = int(object_id)
        return MultiplexBucket(
            self.index,
            self.capacity,
            tuple(existing for existing in self.object_ids if existing != object_id),
        )


@dataclass
class ObjectMultiplexState:
    """Object Multiplex assignment plus per-frame memory metadata."""

    bucket_capacity: int
    buckets: list[MultiplexBucket] = field(default_factory=list)
    object_to_bucket: dict[int, int] = field(default_factory=dict)
    active_object_ids: tuple[int, ...] = ()
    memory: list[TrackMemoryRecord] = field(default_factory=list)
    frame_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bucket_capacity = int(self.bucket_capacity)
        if self.bucket_capacity < 1:
            raise ValueError("ObjectMultiplexState.bucket_capacity must be positive")
        self.buckets = list(self.buckets)
        seen_bucket_indices: set[int] = set()
        derived: dict[int, int] = {}
        ordered_objects: list[int] = []
        for bucket in self.buckets:
            if not isinstance(bucket, MultiplexBucket):
                raise TypeError("ObjectMultiplexState.buckets must contain MultiplexBucket values")
            if bucket.capacity != self.bucket_capacity:
                raise ValueError("ObjectMultiplexState bucket capacity mismatch")
            if bucket.index in seen_bucket_indices:
                raise ValueError("ObjectMultiplexState bucket indices must be unique")
            seen_bucket_indices.add(bucket.index)
            for object_id in bucket.object_ids:
                if object_id in derived:
                    raise ValueError(f"object {object_id} appears in multiple multiplex buckets")
                derived[object_id] = bucket.index
                ordered_objects.append(object_id)
        supplied = {int(k): int(v) for k, v in self.object_to_bucket.items()}
        if supplied and supplied != derived:
            raise ValueError("ObjectMultiplexState.object_to_bucket does not match buckets")
        self.object_to_bucket = derived
        if self.active_object_ids:
            active = tuple(int(object_id) for object_id in self.active_object_ids)
            if set(active) != set(derived):
                raise ValueError("ObjectMultiplexState.active_object_ids does not match buckets")
            self.active_object_ids = active
        else:
            self.active_object_ids = tuple(ordered_objects)
        self.memory = list(self.memory)
        for record in self.memory:
            if not isinstance(record, TrackMemoryRecord):
                raise TypeError("ObjectMultiplexState.memory must contain TrackMemoryRecord values")
            if record.object_id not in self.object_to_bucket:
                raise ValueError("TrackMemoryRecord.object_id is not assigned to a multiplex bucket")
        if self.frame_index is not None:
            self.frame_index = int(self.frame_index)
            if self.frame_index < 0:
                raise ValueError("ObjectMultiplexState.frame_index must be non-negative")

    @classmethod
    def from_object_ids(
        cls,
        object_ids,
        *,
        bucket_capacity: int,
        frame_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ObjectMultiplexState":
        state = cls(bucket_capacity=bucket_capacity, frame_index=frame_index, metadata=metadata or {})
        for object_id in object_ids:
            state.assign_object(int(object_id))
        return state

    def assign_object(self, object_id: int) -> int:
        object_id = int(object_id)
        if object_id < 0:
            raise ValueError("object_id must be non-negative")
        if object_id in self.object_to_bucket:
            return self.object_to_bucket[object_id]
        for i, bucket in enumerate(self.buckets):
            if bucket.has_room:
                self.buckets[i] = bucket.add(object_id)
                self.object_to_bucket[object_id] = bucket.index
                self.active_object_ids = self.active_object_ids + (object_id,)
                return bucket.index
        bucket_index = 0 if not self.buckets else max(bucket.index for bucket in self.buckets) + 1
        bucket = MultiplexBucket(bucket_index, self.bucket_capacity, (object_id,))
        self.buckets.append(bucket)
        self.object_to_bucket[object_id] = bucket_index
        self.active_object_ids = self.active_object_ids + (object_id,)
        return bucket_index

    def remove_object(self, object_id: int) -> None:
        object_id = int(object_id)
        bucket_index = self.object_to_bucket.pop(object_id, None)
        if bucket_index is None:
            return
        self.buckets = [
            bucket.remove(object_id) if bucket.index == bucket_index else bucket
            for bucket in self.buckets
        ]
        self.active_object_ids = tuple(existing for existing in self.active_object_ids if existing != object_id)

    def add_memory(self, record: TrackMemoryRecord) -> None:
        if record.object_id not in self.object_to_bucket:
            raise ValueError("TrackMemoryRecord.object_id is not assigned to a multiplex bucket")
        self.memory.append(record)
        self.frame_index = record.frame_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_capacity": self.bucket_capacity,
            "buckets": [
                {
                    "index": bucket.index,
                    "capacity": bucket.capacity,
                    "object_ids": list(bucket.object_ids),
                }
                for bucket in self.buckets
            ],
            "object_to_bucket": {str(k): v for k, v in self.object_to_bucket.items()},
            "active_object_ids": list(self.active_object_ids),
            "memory": [
                {
                    "object_id": record.object_id,
                    "frame_index": record.frame_index,
                    "mask_shape": None if record.mask_shape is None else list(record.mask_shape),
                    "score": record.score,
                    "metadata": record.metadata,
                }
                for record in self.memory
            ],
            "frame_index": self.frame_index,
            "metadata": self.metadata,
        }
