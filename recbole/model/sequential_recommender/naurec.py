# -*- coding: utf-8 -*-
# @Time   : 2024/03/25
# @Author : Gemini
# @Email  :

r"""
NAURec
################################################

NAURec is a simple additive sequential model that uses a Diagonal Neural Addition Unit (NAU)
to act as a learnable gate. It filters the user's history before aggregating it, allowing
the model to ignore noise items (w=0) or correct signals (w=-1) before summation.

"""

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType, ModelType


class WeightClipper(nn.Module):
    """
    Parametrization that clamps weights to [-1, 1].
    """

    def forward(self, w):
        return torch.clamp(w, -1.0, 1.0)


class NAUSparsityLoss(nn.Module):
    """
    Calculates the NAU sparsity penalty: sum(min(|w|, 1-|w|)).
    Push weights towards {-1, 0, 1}.
    """

    def __init__(self):
        super(NAUSparsityLoss, self).__init__()

    def forward(self, w):
        w_abs = torch.abs(w)
        return torch.sum(torch.min(w_abs, 1.0 - w_abs))


class DiagonalNAU(nn.Module):
    """
    A diagonal NAU layer that learns a scalar weight for every item.
    Acts as a gating mechanism.
    """

    def __init__(self, n_items):
        super(DiagonalNAU, self).__init__()
        self.n_items = n_items
        self.w = nn.Parameter(torch.Tensor(n_items))
        self.reset_parameters()
        parametrize.register_parametrization(self, "w", WeightClipper())

    def reset_parameters(self):
        nn.init.uniform_(self.w, 0, 0.1)

    def forward(self, item_indices):
        return self.w[item_indices].unsqueeze(-1)


class DiagonalNAULayer(nn.Module):
    def __init__(self, n_items, n_dims=4):
        super(DiagonalNAULayer, self).__init__()
        self.n_items = n_items
        self.n_dims = n_dims

        self.w = nn.Parameter(torch.Tensor(n_items, n_dims))
        self.reset_parameters()

        parametrize.register_parametrization(self, "w", WeightClipper())

    def reset_parameters(self):
        nn.init.uniform_(self.w, 0.4, 0.6)

    def forward(self, item_indices):
        return self.w[item_indices]


class DiagoNauLayer(nn.Module):
    """
    A diagonal NAU layer that learns a scalar weight for every item.
    Acts as a gating mechanism.
    """

    def __init__(
        self,
        n_items,
    ):
        super(DiagoNauLayer, self).__init__()
        self.n_items = n_items
        self.w = nn.Parameter(torch.Tensor(n_items))
        self.reset_parameters()
        parametrize.register_parametrization(self, "w", WeightClipper())

    def reset_parameters(self):
        nn.init.uniform_(self.w, 0, 0.1)

    def forward(self, item_indices):
        return self.w[item_indices].unsqueeze(-1)


class NAURec(SequentialRecommender):
    """
    A simple Gated Additive Recommender.
    User Vector = Sum( NAU_Gate(Item) * Embedding(Item) )
    """

    input_type = InputType.PAIRWISE
    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(NAURec, self).__init__(config, dataset)

        self.config = config
        self.loss_type = config["loss_type"]
        self.nau_reg_weight = config["nau_reg_weight"]
        self.dropout_prob = config["dropout_prob"]
        self.nau_gate = DiagonalNAU(self.n_items)
        self.emb_dropout = nn.Dropout(self.dropout_prob)

        self.sparsity_loss_fct = NAUSparsityLoss()

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            if self.config["optimize"]:
                self.loss_fct = nn.BCELoss()
                self.loss_type = "BCE"
            else:
                self.loss_fct = nn.CrossEntropyLoss()

        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, item_seq, item_seq_len):
        # # item_emb = self.item_embedding(item_seq)
        # gate_values = self.nau_gate(item_seq)
        # # gated_emb = item_emb * gate_values
        # # gated_emb = self.emb_dropout(gated_emb)
        # # mask = (item_seq != 0).float().unsqueeze(-1)
        # # gated_emb = gated_emb * mask
        # # user_emb = torch.sum(gated_emb, dim=1)
        # # user_emb = self.LayerNorm(user_emb)
        # scores = torch.zeros((item_seq.size(0), self.n_items))
        # scores.scatter_add_(src=gate_values, index=item_seq, dim=1)

        batch_size, seq_len = item_seq.shape
        gate_values = self.nau_gate(item_seq).squeeze(-1)
        batch_indices = (
            torch.arange(batch_size, device=item_seq.device)
            .unsqueeze(1)
            .expand_as(item_seq)
        )
        indices = torch.stack([batch_indices.reshape(-1), item_seq.reshape(-1)])
        values = gate_values.reshape(-1)
        user_profile_sparse = torch.sparse_coo_tensor(
            indices, values, size=(batch_size, self.n_items)
        ).to_dense()

        return user_profile_sparse

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        logits = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_score = logits.gather(1, pos_items.unsqueeze(1)).squeeze(-1)
            neg_score = logits.gather(1, neg_items.unsqueeze(1)).squeeze(-1)

            task_loss = self.loss_fct(pos_score, neg_score)

        elif self.loss_type == "CE":
            task_loss = self.loss_fct(logits, pos_items)
        elif self.loss_type == "BCE":
            labels = torch.zeros(
                (interaction.length, self.n_items),
                dtype=torch.float,
                device=self.device,
            )
            labels.scatter_add_(index=pos_items, src=(pos_items > 0).float(), dim=1)
            task_loss = self.loss_fct(torch.nn.functional.sigmoid(logits), labels)

        reg_loss = self.sparsity_loss_fct(self.nau_gate.w)

        return task_loss + (self.nau_reg_weight * reg_loss)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        scores = self.forward(item_seq, item_seq_len)

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(item_seq, item_seq_len)

        return seq_output
