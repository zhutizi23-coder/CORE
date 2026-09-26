"""Original token positions/identities, independent of compact KV slots."""

import torch


class CacheMetadata:
    def __init__(self):
        self.reset()

    def reset(self):
        self.token_ids = None
        self.positions = {}
        self.next_position = {}

    def record(self, input_ids, position_ids=None):
        if input_ids is None:
            raise ValueError("CORE token metadata requires input_ids")
        if position_ids is None:
            start = 0 if self.token_ids is None else self.token_ids.shape[1]
            position_ids = torch.arange(start, start + input_ids.shape[1], device=input_ids.device)[None]
        position_ids = position_ids.expand(input_ids.shape[0], -1).long()
        size = int(position_ids.max()) + 1
        previous = self.token_ids
        if previous is not None:
            if previous.shape[0] != input_ids.shape[0]:
                raise ValueError("Batch changed without a sequence reset")
            size = max(size, previous.shape[1])
        ids = input_ids.new_full((input_ids.shape[0], size), -1)
        if previous is not None:
            ids[:, :previous.shape[1]] = previous.to(ids.device)
        ids.scatter_(1, position_ids, input_ids)
        self.token_ids = ids

    def align(self, layer_idx, batch_size, cache_length, device):
        previous = self.positions.get(layer_idx)
        old_length = 0 if previous is None else previous.shape[1]
        if cache_length < old_length:
            raise RuntimeError("KV metadata was not pruned with the cache; reset at new sample boundaries")
        start = self.next_position.get(layer_idx, 0)
        added = cache_length - old_length
        tail = torch.arange(start, start + added, device=device)[None].expand(batch_size, -1)
        positions = tail if previous is None else torch.cat((previous.to(device), tail), dim=1)
        self.positions[layer_idx] = positions
        self.next_position[layer_idx] = start + added
        ids = positions.new_full(positions.shape, -1)
        if self.token_ids is not None:
            journal = self.token_ids.to(device)
            if positions.numel() and int(positions.max()) >= journal.shape[1]:
                raise RuntimeError("Missing input token IDs for newly appended KV entries")
            ids = journal.gather(1, positions)
        return positions, ids, self.next_position[layer_idx]

    def retain(self, layer_idx, indices):
        self.positions[layer_idx] = self.positions[layer_idx].gather(1, indices).contiguous()
