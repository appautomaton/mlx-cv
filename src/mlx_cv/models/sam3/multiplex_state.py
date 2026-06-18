"""Model-side Object Multiplex state for SAM3 video."""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import numpy as np

__all__ = [
    "SAM3MultiplexController",
    "SAM3MultiplexState",
    "PADDING_OBJECT_INDEX",
    "REMOVED_OBJECT_INDEX",
]


PADDING_OBJECT_INDEX = -1
REMOVED_OBJECT_INDEX = -1116


def _as_assignments(assignments: Sequence[Sequence[int]]) -> list[list[int]]:
    out = [[int(v) for v in bucket] for bucket in assignments]
    if not out:
        raise ValueError("SAM3 multiplex state requires at least one bucket")
    width = len(out[0])
    if width <= 0:
        raise ValueError("SAM3 multiplex buckets must be non-empty")
    if any(len(bucket) != width for bucket in out):
        raise ValueError("SAM3 multiplex buckets must all have the same width")
    return out


class SAM3MultiplexState:
    """Fixed-size bucket assignment plus MLX mux/demux matrices.

    Data-space tensors have shape ``(O, ...)`` where ``O`` is the number of
    valid model object indices. Multiplex-space tensors have shape
    ``(num_buckets, multiplex_count, ...)``. Padding and removed slots are
    ignored by demux.
    """

    def __init__(
        self,
        assignments: Sequence[Sequence[int]],
        *,
        allowed_bucket_capacity: int | None = None,
        object_ids: Sequence[int] | None = None,
        dtype=mx.float32,
    ) -> None:
        self.dtype = dtype
        self.allowed_bucket_capacity = int(allowed_bucket_capacity or len(assignments[0]))
        if self.allowed_bucket_capacity <= 0:
            raise ValueError("SAM3 multiplex allowed_bucket_capacity must be positive")
        self._initialize(assignments, object_ids=object_ids)

    def _initialize(self, assignments: Sequence[Sequence[int]], *, object_ids: Sequence[int] | None) -> None:
        self.assignments = _as_assignments(assignments)
        self.num_buckets = len(self.assignments)
        self.multiplex_count = len(self.assignments[0])
        if self.allowed_bucket_capacity > self.multiplex_count:
            raise ValueError("SAM3 multiplex allowed_bucket_capacity cannot exceed multiplex_count")

        valid: list[int] = []
        non_padding = 0
        for bucket in self.assignments:
            bucket_non_padding = sum(1 for obj_idx in bucket if obj_idx != PADDING_OBJECT_INDEX)
            if bucket_non_padding > self.allowed_bucket_capacity:
                raise ValueError("SAM3 multiplex bucket exceeds allowed capacity")
            non_padding += bucket_non_padding
            for obj_idx in bucket:
                if obj_idx >= 0:
                    valid.append(obj_idx)
                elif obj_idx not in (PADDING_OBJECT_INDEX, REMOVED_OBJECT_INDEX):
                    raise ValueError(f"invalid SAM3 multiplex sentinel {obj_idx}")
        if len(set(valid)) != len(valid):
            raise ValueError("SAM3 multiplex model object indices must be unique")
        if sorted(valid) != list(range(len(valid))):
            raise ValueError("SAM3 multiplex model object indices must be contiguous from zero")

        self.total_valid_entries = len(valid)
        self.total_non_padding_entries = non_padding
        if object_ids is None:
            self.object_ids = tuple(range(self.total_valid_entries))
        else:
            self.object_ids = tuple(int(v) for v in object_ids)
            if len(self.object_ids) != self.total_valid_entries:
                raise ValueError("SAM3 multiplex object_ids must map 1:1 to valid entries")
            if len(set(self.object_ids)) != len(self.object_ids):
                raise ValueError("SAM3 multiplex object_ids must be unique")
        self.object_id_to_index = {object_id: i for i, object_id in enumerate(self.object_ids)}
        self._precompute_transition_matrices()

    @property
    def available_slots(self) -> int:
        return self.num_buckets * self.allowed_bucket_capacity - self.total_non_padding_entries

    @property
    def assignments_tuple(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(bucket) for bucket in self.assignments)

    @property
    def object_ids_array(self) -> mx.array:
        return mx.array(np.asarray(self.object_ids, dtype=np.int32))

    @property
    def valid_mask(self) -> mx.array:
        return self.get_valid_object_mask()

    def _precompute_transition_matrices(self) -> None:
        mux = np.zeros(
            (self.num_buckets * self.multiplex_count, self.total_valid_entries),
            dtype=np.float32,
        )
        demux = np.zeros(
            (self.total_valid_entries, self.num_buckets * self.multiplex_count),
            dtype=np.float32,
        )
        for bucket_idx, bucket in enumerate(self.assignments):
            for slot_idx, object_idx in enumerate(bucket):
                if object_idx >= 0:
                    flat_slot = bucket_idx * self.multiplex_count + slot_idx
                    mux[flat_slot, object_idx] = 1.0
                    demux[object_idx, flat_slot] = 1.0
        self.mux_matrix = mx.array(mux, dtype=self.dtype)
        self.demux_matrix = mx.array(demux, dtype=self.dtype)

    def mux(self, x: mx.array) -> mx.array:
        if len(x.shape) < 1:
            raise ValueError("SAM3 multiplex mux input must have at least one axis")
        if int(x.shape[0]) != self.total_valid_entries:
            raise ValueError(
                f"SAM3 mux expected {self.total_valid_entries} object entries, got {int(x.shape[0])}"
            )
        flat = x.reshape(self.total_valid_entries, -1).astype(self.dtype)
        out = self.mux_matrix @ flat
        return out.reshape((self.num_buckets, self.multiplex_count) + tuple(x.shape[1:]))

    def demux(self, x: mx.array) -> mx.array:
        if len(x.shape) < 2:
            raise ValueError("SAM3 multiplex demux input must have bucket and slot axes")
        if int(x.shape[0]) != self.num_buckets or int(x.shape[1]) != self.multiplex_count:
            raise ValueError(
                "SAM3 demux expected "
                f"({self.num_buckets}, {self.multiplex_count}, ...), got {tuple(x.shape)}"
            )
        flat = x.reshape(self.num_buckets * self.multiplex_count, -1).astype(self.dtype)
        out = self.demux_matrix @ flat
        return out.reshape((self.total_valid_entries,) + tuple(x.shape[2:]))

    def get_valid_object_mask(self) -> mx.array:
        return mx.array(
            np.asarray([[obj_idx >= 0 for obj_idx in bucket] for bucket in self.assignments], dtype=np.bool_)
        )

    def get_all_valid_object_idx(self) -> set[int]:
        return {obj_idx for bucket in self.assignments for obj_idx in bucket if obj_idx >= 0}

    def find_next_batch_of_available_indices(
        self,
        num_objects: int,
        *,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> list[int]:
        del prefer_new_buckets
        num_objects = int(num_objects)
        if num_objects <= 0:
            raise ValueError("num_objects must be positive")
        if not allow_new_buckets and self.available_slots < num_objects:
            raise ValueError("not enough SAM3 multiplex slots available")
        start = self.total_valid_entries
        return list(range(start, start + num_objects))

    def add_object_id(self, object_id: int, *, allow_new_buckets: bool = True) -> int:
        new_index = self.total_valid_entries
        self.add_objects([new_index], object_ids=[int(object_id)], allow_new_buckets=allow_new_buckets)
        return new_index

    def add_objects(
        self,
        object_indices: Sequence[int],
        *,
        object_ids: Sequence[int] | None = None,
        allow_new_buckets: bool = True,
        prefer_new_buckets: bool = False,
    ) -> None:
        object_indices = [int(v) for v in object_indices]
        if not object_indices:
            return
        if object_indices != sorted(object_indices):
            raise ValueError("SAM3 multiplex object_indices must be sorted")
        expected = list(range(self.total_valid_entries, self.total_valid_entries + len(object_indices)))
        if object_indices != expected:
            raise ValueError("SAM3 multiplex added object_indices must continue the current sequence")
        if object_ids is None:
            new_object_ids = tuple(object_indices)
        else:
            new_object_ids = tuple(int(v) for v in object_ids)
            if len(new_object_ids) != len(object_indices):
                raise ValueError("SAM3 multiplex added object_ids length mismatch")
        if set(new_object_ids) & set(self.object_ids):
            raise ValueError("SAM3 multiplex added object_ids must be new")

        assignments = [bucket.copy() for bucket in self.assignments]
        remaining = list(object_indices)
        if not prefer_new_buckets:
            for bucket in assignments:
                for slot_idx in range(self.allowed_bucket_capacity):
                    if bucket[slot_idx] == PADDING_OBJECT_INDEX:
                        bucket[slot_idx] = remaining.pop(0)
                        if not remaining:
                            break
                if not remaining:
                    break
        if remaining and not allow_new_buckets:
            raise ValueError("SAM3 multiplex cannot place objects without creating new buckets")
        while remaining:
            bucket = [PADDING_OBJECT_INDEX] * self.multiplex_count
            for slot_idx in range(self.allowed_bucket_capacity):
                if not remaining:
                    break
                bucket[slot_idx] = remaining.pop(0)
            assignments.append(bucket)
        self._initialize(assignments, object_ids=self.object_ids + new_object_ids)

    def remove_object_id(self, object_id: int, *, strict: bool = True) -> list[int]:
        object_id = int(object_id)
        if object_id not in self.object_id_to_index:
            if strict:
                raise ValueError(f"SAM3 multiplex object_id is not active: {object_id}")
            return list(range(self.num_buckets))
        return self.remove_objects([self.object_id_to_index[object_id]], strict=strict)

    def remove_objects(self, object_indices: Sequence[int], *, strict: bool = True) -> list[int]:
        pending = {int(v) for v in object_indices}
        assignments = [bucket.copy() for bucket in self.assignments]
        for bucket in assignments:
            for slot_idx, obj_idx in enumerate(bucket):
                if obj_idx in pending:
                    bucket[slot_idx] = REMOVED_OBJECT_INDEX
                    pending.remove(obj_idx)
        if pending and strict:
            raise ValueError(f"SAM3 multiplex object indices not found: {sorted(pending)}")

        kept = [
            bucket
            for bucket in assignments
            if any(obj_idx not in (PADDING_OBJECT_INDEX, REMOVED_OBJECT_INDEX) for obj_idx in bucket)
        ]
        if not kept:
            kept = [[PADDING_OBJECT_INDEX] * self.multiplex_count]

        positive = sorted({obj_idx for bucket in kept for obj_idx in bucket if obj_idx >= 0})
        old_object_ids = self.object_ids
        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(positive)}
        for bucket in kept:
            for slot_idx, obj_idx in enumerate(bucket):
                if obj_idx >= 0:
                    bucket[slot_idx] = remap[obj_idx]
        new_object_ids = tuple(old_object_ids[old_idx] for old_idx in positive)
        self._initialize(kept, object_ids=new_object_ids)
        return [i for i, bucket in enumerate(kept) if any(obj_idx >= 0 for obj_idx in bucket)]

    def capture_taps(self) -> dict[str, object]:
        probe = mx.arange(self.total_valid_entries, dtype=mx.float32).reshape(self.total_valid_entries, 1)
        mux_probe = self.mux(probe)
        return {
            "multiplex.assignments": [bucket.copy() for bucket in self.assignments],
            "multiplex.valid_mask": self.get_valid_object_mask(),
            "multiplex.object_ids": self.object_ids_array,
            "multiplex.mux_probe": mux_probe,
            "multiplex.demux_probe": self.demux(mux_probe),
        }


class SAM3MultiplexController:
    """Deterministic single-device bucket planner for SAM3 video inference."""

    def __init__(self, multiplex_count: int, *, eval_multiplex_count: int | None = None) -> None:
        self.multiplex_count = int(multiplex_count)
        self.eval_multiplex_count = int(eval_multiplex_count or multiplex_count)
        if self.multiplex_count <= 0 or self.eval_multiplex_count <= 0:
            raise ValueError("SAM3 multiplex counts must be positive")
        if self.eval_multiplex_count > self.multiplex_count:
            raise ValueError("SAM3 eval_multiplex_count cannot exceed multiplex_count")

    @property
    def allowed_bucket_capacity(self) -> int:
        return self.eval_multiplex_count

    def get_state(
        self,
        num_valid_entries: int,
        *,
        object_ids: Sequence[int] | None = None,
        dtype=mx.float32,
    ) -> SAM3MultiplexState:
        num_valid_entries = int(num_valid_entries)
        if num_valid_entries < 0:
            raise ValueError("num_valid_entries must be non-negative")
        num_buckets = max(1, math.ceil(max(num_valid_entries, 1) / self.allowed_bucket_capacity))
        assignments: list[list[int]] = []
        next_object_idx = 0
        for _ in range(num_buckets):
            bucket = [PADDING_OBJECT_INDEX] * self.multiplex_count
            for slot_idx in range(self.allowed_bucket_capacity):
                if next_object_idx >= num_valid_entries:
                    break
                bucket[slot_idx] = next_object_idx
                next_object_idx += 1
            assignments.append(bucket)
        return SAM3MultiplexState(
            assignments,
            allowed_bucket_capacity=self.allowed_bucket_capacity,
            object_ids=object_ids,
            dtype=dtype,
        )
