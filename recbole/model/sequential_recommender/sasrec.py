# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from recbole.model.loss import sigmoid_focal_loss, BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import ItemBias
from recbole.utils.utils import batched_isin
from recbole.utils import InputType


class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = self.hidden_size * 4  # config["inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.repeat_loss_scale = config["repeat_loss_scale"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.item_bias = ItemBias(dataset=dataset)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # --- CHANGED: Use PyTorch native TransformerEncoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.hidden_dropout_prob,
            activation=self.hidden_act if isinstance(self.hidden_act, str) else "gelu",
            layer_norm_eps=self.layer_norm_eps,
            batch_first=True
        )
        self.trm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            if config["optimize"]:
                self.loss_fct = nn.BCEWithLogitsLoss()
            else:
                self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def make_basket_causal_mask(self, basket_indices):
        query_time = basket_indices.unsqueeze(2)
        key_time = basket_indices.unsqueeze(1)
        causal_mask = key_time > query_time
        return causal_mask.repeat_interleave(self.n_heads, dim=0)

    def forward(self, item_seq, item_seq_len, time_seq):
        time_seq -= time_seq.min(dim=1).values.unsqueeze(1)
        position_embedding = self.position_embedding(time_seq.long())
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # CHANGED: Use custom causal mask and standard padding mask
        causal_mask = self.make_basket_causal_mask(time_seq.long())
        pad_mask = (item_seq == 0)

        trm_output = self.trm_encoder(
            src=input_emb,
            mask=causal_mask,
            src_key_padding_mask=pad_mask
        )

        output = trm_output[:, -1, :]
        return F.normalize(output, p=2, dim=-1)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.NEG_ITEM_ID in interaction:
            neg_items = interaction[self.NEG_ITEM_ID]
            n_neg = neg_items.shape[2] if neg_items.dim() > 2 else 1
            neg_items = neg_items.view(interaction.length, self.max_seq_length * n_neg)

            # Normalization mimicking `get_fused_target_embedding` from CAT
            neg_emb = F.normalize(self.item_embedding(neg_items), p=2, dim=-1)
            neg_emb = neg_emb.view(interaction.length, self.max_seq_length * n_neg, -1)

            pos_emb = F.normalize(self.item_embedding(pos_items), p=2, dim=-1)

            pos_bias = self.item_bias(pos_items)
            neg_bias = self.item_bias(neg_items).view(interaction.length, self.max_seq_length * n_neg)

            pos_logits = (seq_output.unsqueeze(1) * pos_emb).sum(-1) * self.logit_scale.exp() + pos_bias
            neg_logits = (seq_output.unsqueeze(1) * neg_emb).sum(-1) * self.logit_scale.exp() + neg_bias
            logits = torch.cat([pos_logits, neg_logits], dim=-1)

            labels = torch.tensor([1.0] * self.max_seq_length + [0.0] * n_neg * self.max_seq_length,
                                  device=self.device).expand_as(logits)

            raw_loss = sigmoid_focal_loss(logits, labels, reduction='none')
            valid_mask = (pos_items > 0).repeat(1, n_neg + 1)
            masked_loss = raw_loss * valid_mask.float()

            return masked_loss.sum() / (valid_mask.sum() * 2 + 1e-9)

        all_items = torch.arange(self.n_items, device=self.device)
        all_emb = F.normalize(self.item_embedding(all_items), p=2, dim=-1)
        logits = torch.matmul(seq_output, all_emb.transpose(1, 0))
        logits *= self.logit_scale.exp().clamp(max=100)
        logits += self.item_bias()

        labels = torch.zeros_like(logits).scatter_add(
            dim=1, index=pos_items, src=torch.ones_like(pos_items).float()
        )
        labels[:, 0] = 0
        raw_loss = sigmoid_focal_loss(logits, labels, reduction='none')
        repeat_mask = batched_isin(pos_items, interaction[self.ITEM_SEQ])

        scale_values = torch.where(
            repeat_mask,
            torch.tensor(self.repeat_loss_scale, device=self.device, dtype=torch.float),
            torch.tensor(1.0, device=self.device, dtype=torch.float)
        )
        loss_weights = torch.ones_like(raw_loss)
        loss_weights.scatter_(dim=1, index=pos_items, src=scale_values)
        scaled_loss = raw_loss * loss_weights

        return scaled_loss.mean()

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len, time_seq)

        test_item_emb = F.normalize(self.item_embedding(test_item), p=2, dim=-1)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        scores *= self.logit_scale.clamp(max=100)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        seq_output = self.forward(item_seq, item_seq_len, time_seq)

        all_items = torch.arange(self.n_items, device=self.device)
        test_items_emb = F.normalize(self.item_embedding(all_items), p=2, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        scores *= self.logit_scale.clamp(max=100)
        return scores