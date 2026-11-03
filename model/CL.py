import torch
import torch.nn as nn
import torch.nn.functional as F


class CL(nn.Module):

    def __init__(self, temperature=0.1, neg_sample_ratio=50,
                 temp_annealing=0.99, temp_annealing_step=40, eps=1e-8):
        super(CL, self).__init__()
        # temperature=0.1 — temperature parameter controlling the "steepness" of
        # the similarity distribution. Higher is looser, lower is stricter.
        self.temperature = nn.Parameter(torch.tensor(temperature))
        # neg_sample_ratio=50 — how many negatives to pair with each positive.
        # It decides how many negatives K are drawn from the negative pool per sample.
        self.neg_sample_ratio = neg_sample_ratio
        # temp_annealing=0.99 — annealing factor; every temp_annealing_step steps
        # the temperature is multiplied by it, approaching 0.01 in the limit.
        self.temp_annealing = temp_annealing
        self.eps = eps
        # temp_annealing_step — annealing period, i.e. how many steps between updates.
        self.temp_annealing_step = temp_annealing_step

        self.register_buffer('temp_update_count', torch.tensor(0))

    def forward(self, z_i, z_j):
        """
        Align the GCN outputs of the two LLMs on the same branch (spa or fea).
        z_i, z_j: [B, D]  latents of the two LLMs within the same batch.
        """
        batch_size = z_i.size(0)

        # 1. L2 normalization
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # 2. Positive similarity (two LLM latents of the same spot)
        pos_sim = torch.sum(z_i * z_j, dim=1)

        # 3. Build the negative pool [2B, D]
        all_neg_pool = torch.cat([z_i, z_j], dim=0)

        # 4. Sample K negatives for each sample
        neg_indices = self._sample_negatives(batch_size, total_size=2 * batch_size)
        z_neg = all_neg_pool[neg_indices]  # [B, K, D]

        # 5. Negative similarity
        z_i_exp = z_i.unsqueeze(1)  # [B, 1, D]
        neg_sim = torch.sum(z_i_exp * z_neg, dim=-1)  # [B, K]

        # 6. logits / labels
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)

        # 7. InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        # 8. Temperature annealing
        if self.training and self.temp_annealing < 1.0:
            self.temp_update_count += 1
            if self.temp_update_count % self.temp_annealing_step == 0:
                # .temperature.data is the tensor of the temperature parameter;
                # *= updates it in place.
                # self.temp_annealing=0.99 — annealing factor, multiplied every
                # 100 steps, approaching 0.01 in the limit.
                self.temperature.data *= self.temp_annealing

        return loss

    def _sample_negatives(self, batch_size, total_size):
        """Sample K negatives for each i, excluding i itself and its positive."""
        device = self.temperature.device
        num_negs = min(total_size - 2, self.neg_sample_ratio)
        all_indices = torch.arange(total_size, device=device)
        neg_indices = []
        for i in range(batch_size):
            exclude = torch.tensor([i, i + batch_size], device=device)
            possible = all_indices[~torch.isin(all_indices, exclude)]
            selected = possible[torch.randperm(possible.size(0),
                                                device=device)[:num_negs]]
            neg_indices.append(selected)
        return torch.stack(neg_indices)
