import torch


class KVCache:
    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: str | torch.device,
    ):
        self._max_seq_len = max_seq_len
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._device = device
        self._keys = None
        self._values = None
        self._seq_len = 0

    def update_and_get_keys(self, k: torch.Tensor):
        if self._keys is None:
            batch_size = k.shape[0]
            self._keys = torch.zeros(
                batch_size,
                self._num_heads,
                self._max_seq_len,
                self._head_dim,
                device=self._device,
                dtype=k.dtype,
            )
        seq_len = k.shape[2]
        self._keys[:, :, self._seq_len : self._seq_len + seq_len] = k
        return self._keys[:, :, : self._seq_len + seq_len]

    def update_and_get_values(self, v: torch.Tensor):
        if self._values is None:
            batch_size = v.shape[0]
            self._values = torch.zeros(
                batch_size,
                self._num_heads,
                self._max_seq_len,
                self._head_dim,
                device=self._device,
                dtype=v.dtype,
            )
        seq_len = v.shape[2]
        self._values[:, :, self._seq_len : self._seq_len + seq_len] = v
        self._seq_len += seq_len
        return self._values[:, :, : self._seq_len]

    def get_position_offset(self) -> int:
        return self._seq_len

    def evict_range(self, start: int, count: int) -> None:
        if self._keys is None or count <= 0:
            return
        end = start + count
        tail_len = self._seq_len - end
        if tail_len > 0:
            self._keys[:, :, start : start + tail_len] = self._keys[
                :, :, end : self._seq_len
            ].clone()
            self._values[:, :, start : start + tail_len] = self._values[
                :, :, end : self._seq_len
            ].clone()
        self._seq_len -= count


class GradSafeKVCache:
    """KV cache using list-based concatenation instead of in-place buffer writes.

    Standard KVCache uses in-place assignment (buffer[:, :, start:end] = k) which
    breaks autograd when gradients need to flow through cached K/V tensors. This
    implementation appends to a list and concatenates, producing new tensors each
    time so autograd graphs remain valid.
    """

    def __init__(self) -> None:
        self._keys_list: list[torch.Tensor] = []
        self._values_list: list[torch.Tensor] = []
        self._seq_len: int = 0

    @classmethod
    def from_standard(cls, kv_cache: KVCache) -> "GradSafeKVCache":
        gc = cls()
        if kv_cache._keys is not None and kv_cache._seq_len > 0:
            gc._keys_list = [kv_cache._keys[:, :, : kv_cache._seq_len].detach()]
            gc._values_list = [kv_cache._values[:, :, : kv_cache._seq_len].detach()]
            gc._seq_len = kv_cache._seq_len
        return gc

    def update_and_get_keys(self, k: torch.Tensor) -> torch.Tensor:
        self._keys_list.append(k)
        return torch.cat(self._keys_list, dim=2)

    def update_and_get_values(self, v: torch.Tensor) -> torch.Tensor:
        self._values_list.append(v)
        self._seq_len += v.shape[2]
        return torch.cat(self._values_list, dim=2)

    def get_position_offset(self) -> int:
        return self._seq_len

    def evict_range(self, start: int, count: int) -> None:
        if count <= 0:
            return
        end = start + count
        k = torch.cat(self._keys_list, dim=2)
        v = torch.cat(self._values_list, dim=2)
        k = torch.cat([k[:, :, :start], k[:, :, end:]], dim=2)
        v = torch.cat([v[:, :, :start], v[:, :, end:]], dim=2)
        self._keys_list = [k]
        self._values_list = [v]
        self._seq_len -= count

    def freeze(self) -> None:
        """Detach and consolidate all cached K/V into a single tensor."""
        if self._keys_list:
            self._keys_list = [torch.cat(self._keys_list, dim=2).detach()]
            self._values_list = [torch.cat(self._values_list, dim=2).detach()]
