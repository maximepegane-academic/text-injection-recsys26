# -*- coding: utf-8 -*-
# @Time   : 2024/03/25 10:34
# @Author : Maxime Pegane
# @Email  :

r"""
DREAM
################################################

Reference:
    Feng Yu*, Qiang Liu* et al. "A Dynamic Recurrent Model for Next Basket Recommendation." in SIGIR’16.
* = equal contribution
"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType, ModelType


class DREAM(SequentialRecommender):
    r"""DREAM is a model that incorporate RNN for recommendation and uses an item embedding aggregation to obtain a
    basket level representation, which is then fed to a ."""

    input_type = InputType.PAIRWISE
    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(DREAM, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["embedding_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.aggregation = config["aggregation"]
        self.non_linearity = config["non_linearity"]
        self.rnn_type = config["rnn_type"]
        self.max_seq_len = config["MAX_ITEM_LIST_LENGTH"]
        self.eps = 1e-10

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)

        if config["aggregation"] == "mean":
            self.aggregation = "mean"
        else:
            self.aggregation = "amax"

        recurrent_network = None
        if self.rnn_type == "LSTM":
            recurrent_network = nn.LSTM
        elif self.rnn_type == "RNN":
            recurrent_network = nn.RNN

        non_linearity = self.non_linearity

        self.recurrent_network = recurrent_network(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_prob,
            batch_first=True,
        )

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
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        if isinstance(module, nn.RNNBase):
            # same init as original research paper
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    @staticmethod
    def compute_flattened_index(adjusted_time_seq, embedding_size, max_seq_len):
        """
         function to compute indexes needed to aggregate item embedding along each basket (ending with N aggregated
         embedding, with N being the number of basket in user's history)
         Args:
            - adjusted_time_seq: the sequence of purchases of size (N, max_seq_len)
            - embedding_size: size of of the item embeddings
            - max_seq_len: size of the interaction history to take into accout
        Returns:
            - a flattened list of index to use for a torch.scatter_reduce function
        """
        batch_size = adjusted_time_seq.size(0)
        users = torch.arange(batch_size, device=adjusted_time_seq.device).view(
            -1, 1, 1
        )  # [B, 1, 1]
        times = adjusted_time_seq.view(batch_size, -1, 1)  # [B, S, 1]
        emb_idx = torch.arange(embedding_size, device=adjusted_time_seq.device).view(
            1, 1, -1
        )  # [1, 1, E]
        flattened_idx = (
            (
                (users * max_seq_len * embedding_size)  # User offset
                + (times * embedding_size)  # Time offset
                + emb_idx.expand(batch_size, max_seq_len, -1)  # Embedding index
            )
            .flatten()
            .long()
        )

        return flattened_idx

    @staticmethod
    def scatter_aggregate(
        embeddings, flattened_index, aggregation, max_seq_len, embedding_size
    ):
        """
        Uses a flat list of index to calculate the max/mean of each item in a basket and returns a list of padded embeddings
        (the returned "embeddings" after the N found baskets are vectors of 0 to fill the matrix of shape
        (batch_size, #_of_bakset, embedding_size))
        Args:
            - embeddings: batched tensor matrix of item embeddings
            - flattened_index: flat tensor of index to aggregate each item's embeddings into basket embedding
            - aggregation: one of {'amax', 'mean'} for the torch.scatter_reduce_ reduce argument
            - embedding_size: size of of the item embeddings
            - max_seq_len: size of the interaction history to take into accout
        Returns:
            - A batched set of matrices that for each users contains N aggregated embedding
        """
        batch_size = embeddings.size(0)
        out_size = batch_size * max_seq_len * embedding_size

        aggregated = torch.zeros(
            out_size, dtype=embeddings.dtype, device=embeddings.device
        )
        aggregated.scatter_reduce_(
            0,
            flattened_index,
            embeddings.contiguous().view(-1),
            reduce=aggregation,
            include_self=False,
        )

        return aggregated.view(batch_size, max_seq_len, embedding_size)

    def forward(self, item_seq, time_seq, seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        relative_time = time_seq.max(dim=1, keepdim=True).values - time_seq

        adjusted_time_seq = torch.where(
            time_seq > 0,
            (self.max_seq_len - 1 - relative_time).clamp(min=1),
            torch.zeros_like(time_seq),
        ).long()
        flattened_index = self.compute_flattened_index(
            adjusted_time_seq=adjusted_time_seq,
            embedding_size=self.embedding_size,
            max_seq_len=self.max_seq_len,
        )

        aggregated_baskets = self.scatter_aggregate(
            embeddings=item_seq_emb_dropout,
            flattened_index=flattened_index,
            aggregation=self.aggregation,
            max_seq_len=self.max_seq_len,
            embedding_size=self.embedding_size,
        )

        recurrent_output, _ = self.recurrent_network(aggregated_baskets)
        seq_output = recurrent_output[:, -1, :]
        return seq_output

    def calculate_loss(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_FIELD + "_list"]

        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, time_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items_emb = self.item_embedding(neg_items)
            pos_items_emb = self.item_embedding(pos_items)

            if pos_items.dim() == 1:
                neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
                pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)

            else:
                neg_score = torch.sum(seq_output.unsqueeze(1) * neg_items_emb, dim=-1)
                pos_score = torch.sum(seq_output.unsqueeze(1) * pos_items_emb, dim=-1)

                pos_mask = (pos_items > 0).float()
                loss = -nn.functional.logsigmoid(pos_score - neg_score)
                loss = (loss * pos_mask).sum(1)
                return loss.mean()
                # pos_score = pos_score[pos_items > 0]
                # neg_score = neg_score[pos_items > 0]

            loss = self.loss_fct(pos_score, neg_score)
            return loss

        elif self.loss_type == "CE":
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            if pos_items.dim() == 1:
                loss = self.loss_fct(logits, pos_items)
            else:
                labels = torch.zeros_like(logits).scatter_add(
                    dim=1, index=pos_items, src=(pos_items > 0).float()
                )
                loss = self.loss_fct(logits, labels)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_FIELD + "_list"]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, time_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_FIELD + "_list"]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, time_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
