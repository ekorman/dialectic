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
