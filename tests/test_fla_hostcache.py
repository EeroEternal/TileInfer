"""The GDN batch-meta host cache (``tileinfer.vllm.ops.fla.hostcache``).

§37.2: the chunked GDN prefill chain converts ``cu_seqlens`` / ``chunk_indices`` back to the host
about seven times per layer, and those tensors describe the **batch**, not the layer -- ~210
device->host synchronisations per prefill step for identical answers.  The cache removes the repeats.

What must hold, and is pinned here:

* the values are exactly what ``.tolist()`` produced (an int list / a list of int pairs);
* a repeat call on the *same* tensor reuses the cached value, a different tensor does not;
* ``TILEINFER_GDN_HOST_CACHE=0`` restores "reconverted on every call";
* the cache keeps a strong reference to the source, so a freed tensor cannot have its address
  recycled into a stale hit.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.vllm.ops.fla import hostcache


def setup_function(_fn):
    hostcache.reset_cache()


@pytest.fixture()
def caching(monkeypatch):
    """The cache is **opt-in** now (measured 0.999, §42), so every caching test switches it on."""
    monkeypatch.setenv("TILEINFER_GDN_HOST_CACHE", "1")
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    hostcache.reset_cache()
    yield
    hostcache.reset_cache()


def test_int_list_matches_tolist(caching):
    t = torch.tensor([0, 5, 9, 12], dtype=torch.int32)
    assert hostcache.int_list(t) == [0, 5, 9, 12]
    assert all(isinstance(v, int) for v in hostcache.int_list(t))


def test_repeat_call_reuses_the_cached_value(caching, monkeypatch):
    t = torch.tensor([0, 3, 7], dtype=torch.int32)
    first = hostcache.int_list(t)
    second = hostcache.int_list(t)
    assert first is second, "the same tensor must hit the cache, not rebuild"


def test_different_tensor_is_not_a_cache_hit(caching, monkeypatch):
    a = torch.tensor([0, 3, 7], dtype=torch.int32)
    b = torch.tensor([0, 3, 7], dtype=torch.int32)
    assert hostcache.int_list(a) == hostcache.int_list(b)
    assert hostcache.int_list(a) is not hostcache.int_list(b)


def test_switch_off_reconverts_every_call(monkeypatch):
    monkeypatch.setenv("TILEINFER_GDN_HOST_CACHE", "0")
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    t = torch.tensor([0, 2, 4], dtype=torch.int32)
    assert hostcache.int_list(t) == [0, 2, 4]
    assert hostcache.int_list(t) is not hostcache.int_list(t), (
        "with the switch off every call must reconvert (the historical behaviour)"
    )


def test_pairs_match_tolist_of_a_2d_table(caching):
    t = torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.int32)
    assert hostcache.pairs_int_list(t) == [(0, 0), (1, 0), (1, 1)]


def test_cache_holds_the_source_alive(caching, monkeypatch):
    """A strong reference is the safety argument: the key can never alias a recycled address."""
    monkeypatch.setenv("TILEINFER_FLAG_WATCH", "0")
    t = torch.tensor([0, 1], dtype=torch.int32)
    hostcache.int_list(t)
    held = [value[1] for value in hostcache._CACHE.values()]
    assert any(item is t for item in held)


def test_reset_cache_empties_it(caching):
    t = torch.tensor([0, 1], dtype=torch.int32)
    hostcache.int_list(t)
    assert hostcache._CACHE
    hostcache.reset_cache()
    assert not hostcache._CACHE
