# -*- coding: utf-8 -*-
# @Time    : 2024
# @Author  : AI Assistant

"""
SimpleDecay
################################################
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType, ModelType


class SimpleDecay(SequentialRecommender):
    r"""
    SimpleDecay weights a user's historical items based on recency before aggregating them.
    Supports three decay types: 'exp' (Exponential), 'hyp' (Hyperbolic), and 'emb' (Learnable Embedding).
    """

    input_type = InputType.POINTWISE
    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SimpleDecay, self).__init__(config, dataset)

        self.decay_type = config.get("decay_type", "exp").lower()

        # Max time difference for the embedding table to prevent out-of-bounds errors.

        if self.decay_type == "exp":
            self.alpha = nn.Parameter(torch.tensor(0.0))
        elif self.decay_type == "hyp":
            self.alpha = nn.Parameter(torch.tensor(1.0))
        elif self.decay_type == "emb":
            self.time_emb = nn.Embedding(self.max_seq_length + 1, 1, padding_idx=0)
            nn.init.normal_(self.time_emb.weight, 0, 0.01)
            #nn.init.constant_(self.time_emb.weight, 1.0)  # Initialize with uniform weight
        else:
            raise ValueError("decay_type must be 'exp', 'hyp', or 'emb'")

        # Use the same standard logit scale as your other models to expand the BCE margin
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_decay_weights(self, item_seq, time_seq):
        """Calculates the time decay weight for each item in the sequence."""
        mask = (item_seq > 0)

        max_time = torch.max(time_seq, dim=1, keepdim=True).values

        time_delta = ((max_time - time_seq).float() + 1) * mask.long()

        if self.decay_type == "exp":
            weights = torch.exp(-(1 + torch.abs(self.alpha)) * time_delta)

        elif self.decay_type == "hyp":
            weights = 1.0 / (1.0 + torch.abs(self.alpha) * time_delta)

        elif self.decay_type == "emb":
            clamped_delta = time_delta.long().clamp(max=self.max_seq_length)
            weights = self.time_emb(clamped_delta).squeeze(-1)

        return weights * mask.float()

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        pos_items = interaction[self.ITEM_ID]

        weights = self.get_decay_weights(item_seq, time_seq)

        # Aggregate the weights for all items in the vocabulary
        batch_size = interaction.length
        scores = torch.zeros((batch_size, self.n_items), device=self.device, dtype=torch.float)
        scores = scores.scatter_add(dim=1, index=item_seq, src=weights)

        # Apply fixed offset (-0.5) to push unseen items to negative logits, then scale
        logits = (scores - 0.5) * self.logit_scale.exp().clamp(max=100)

        # --- MULTI-LABEL BCE SETUP ---
        pos_items_idx = pos_items.unsqueeze(1) if pos_items.dim() == 1 else pos_items

        labels = torch.zeros_like(logits).scatter_add(
            dim=1, index=pos_items_idx, src=torch.ones_like(pos_items_idx).float()
        )
        labels[:, 0] = 0.0  # Ignore padding index

        # Calculate binary cross entropy
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        test_item = interaction[self.ITEM_ID]

        weights = self.get_decay_weights(item_seq, time_seq)

        # Filter weights to only include the test item, then sum them
        match = (item_seq == test_item.unsqueeze(1)).float()
        scores = (match * weights).sum(dim=1)

        # Match training transformation
        logits = (scores - 0.5) * self.logit_scale.exp().clamp(max=100)
        return logits

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]

        weights = self.get_decay_weights(item_seq, time_seq)

        # Aggregate weights across the whole vocabulary
        batch_size = interaction.length
        scores = torch.zeros((batch_size, self.n_items), device=self.device, dtype=torch.float)
        scores = scores.scatter_add(dim=1, src=weights, index=item_seq)

        # Match training transformation
        logits = (scores - 0.5) * self.logit_scale.exp().clamp(max=100)
        return logits