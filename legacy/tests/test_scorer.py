"""Normalize / compare edge cases. SPEC §3."""

from __future__ import annotations

from datetime import datetime

from bson import Decimal128, Int64, ObjectId

from hx.scorer import _equal, normalize


def test_decimal_vs_float():
    a = [{"v": Decimal128("1.23456")}]
    b = [{"v": 1.2346}]  # rounded to 4 dp
    assert normalize(a) == normalize(b)
    assert _equal(a, b, ordered=True)


def test_int64_vs_int():
    a = [{"n": Int64(42)}]
    b = [{"n": 42}]
    assert _equal(a, b, ordered=True)


def test_objectid_and_datetime():
    oid = ObjectId("507f1f77bcf86cd799439011")
    dt = datetime(2015, 1, 2, 3, 4, 5)
    a = [{"_id": oid, "t": dt}]
    b = [{"_id": str(oid), "t": datetime(2015, 1, 2, 3, 4, 5)}]
    assert _equal(a, b, ordered=True)


def test_key_order_irrelevant():
    a = [{"b": 1, "a": 2}]
    b = [{"a": 2, "b": 1}]
    assert normalize(a) == normalize(b)


def test_ordered_vs_unordered():
    a = [{"x": 1}, {"x": 2}]
    b = [{"x": 2}, {"x": 1}]
    assert _equal(a, b, ordered=False)
    assert not _equal(a, b, ordered=True)


def test_extra_fields_fail():
    a = [{"title": "A", "year": 2000}]
    b = [{"title": "A"}]
    assert not _equal(a, b, ordered=True)


def test_empty_result():
    assert normalize([]) == []
    assert _equal([], [], ordered=True)
