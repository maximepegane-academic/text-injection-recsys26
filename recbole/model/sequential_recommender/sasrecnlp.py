# -*- coding: utf-8 -*-
# @Time    : 2026

"""
SASRecNLP
################################################

An extension of SASRec that incorporates NLP embeddings from a HuggingFace model.
The model concatenates the ID embedding and a projected NLP embedding, feeding
a combined representation of size 2*hidden_size into the Transformer encoder.
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from recbole.model.loss import sigmoid_focal_loss, BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import ItemBias
from recbole.utils import build_text_col


class SASRecNLP(SequentialRecommender):
    r"""
    SASRecNLP leverages the self-attentive mechanism of SASRec while
    enriching item representations with text embeddings extracted from a pre-trained NLP model.
    """

    def __init__(self, config, dataset):
        super(SASRecNLP, self).__init__(config, dataset)

        if hasattr(dataset, 'text_field') is False or dataset.text_field is None:
            build_text_col(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # ID embedding size and nlp projection dimension
        self.inner_size = self.hidden_size * 4  # dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.hf_name = config["hugging_face_model"]
        self.text_field = dataset.text_field

        self.item_bias = ItemBias(dataset=dataset)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.gate_temp = nn.Parameter(torch.tensor(2.3))
        self.input_fusion_layer = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.gate_layer = nn.Linear(self.hidden_size * 2, 1)

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.nlp_proj = None
        if self.hf_name:
            hf_config = AutoConfig.from_pretrained(self.hf_name)
            self.hf_hidden_size = hf_config.hidden_size
            self.nlp_proj = nn.Linear(self.hf_hidden_size, self.hidden_size, bias=False)

        if self.train:
            self.build_nlp_lookup_table(dataset)
        else:
            self.text_embedding = nn.Embedding(self.n_items, getattr(self, 'hf_hidden_size', self.hidden_size),
                                               padding_idx=0)

        self.text_embedding.weight.requires_grad = False

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
            if config.get("optimize", False):
                self.loss_fct = nn.BCEWithLogitsLoss()
            else:
                self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if hasattr(self, 'text_embedding') and module is self.text_embedding:
            return

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @torch.no_grad()
    def build_nlp_lookup_table(self, dataset):
        if dataset.config.get("inference", False):
            self.text_embedding = nn.Embedding(self.n_items, self.hf_hidden_size, padding_idx=0, device=self.device)
            return

        if self.hf_name:
            tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
            encoder = AutoModel.from_pretrained(self.hf_name).to(self.device)
            encoder.eval()

            if self.text_field and self.text_field in dataset.item_feat.columns:
                item_texts = dataset.item_feat.sort(self.ITEM_ID)[self.text_field].to_list()
            else:
                item_texts = [""] * self.n_items

            item_texts[0] = ""  # Padding token
            batch_size = 128
            embs = []

            for i in tqdm(range(0, self.n_items, batch_size), desc="Encoding NLP features"):
                batch_texts = item_texts[i: i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                out = encoder(**inputs).last_hidden_state
                embs.append(out[:, 0, :].cpu())

            self.text_embedding = nn.Embedding(self.n_items, self.hf_hidden_size, padding_idx=0, device=self.device)
            self.text_embedding.weight.data.copy_(F.normalize(torch.cat(embs, dim=0), p=2, dim=-1))

            del encoder
            torch.cuda.empty_cache()

    def make_basket_causal_mask(self, basket_indices):
        query_time = basket_indices.unsqueeze(2)
        key_time = basket_indices.unsqueeze(1)
        causal_mask = key_time > query_time
        return causal_mask.repeat_interleave(self.n_heads, dim=0)

    def get_fused_target_embedding(self, item_ids):
        id_emb = self.item_embedding(item_ids)
        text_emb = self.text_embedding(item_ids)

        if self.nlp_proj is not None:
            text_emb = self.nlp_proj(text_emb)

        id_emb = F.normalize(id_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)

        cat_emb = torch.cat([id_emb, text_emb.detach()], dim=-1)
        cat_emb = cat_emb * self.gate_temp.exp().clamp(100, -100)
        modality_weight = torch.sigmoid(self.gate_layer(cat_emb))
        fused_emb = (modality_weight * text_emb) + ((1.0 - modality_weight) * id_emb)

        return F.normalize(fused_emb, p=2, dim=-1)

    def forward(self, item_seq, item_seq_len, time_seq):
        time_seq -= time_seq.min(dim=1).values.unsqueeze(1)
        position_embedding = self.position_embedding(time_seq.long())

        id_emb = self.item_embedding(item_seq)
        text_emb = self.text_embedding(item_seq)
        if self.nlp_proj is not None:
            text_emb = self.nlp_proj(text_emb)

        # id_emb = F.normalize(id_emb, p=2, dim=-1)
        # text_emb = F.normalize(text_emb, p=2, dim=-1)

        cat_input = torch.cat([id_emb, text_emb], dim=-1)
        fused_input = self.input_fusion_layer(cat_input)

        input_emb = fused_input + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

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

            neg_emb = F.normalize(self.get_fused_target_embedding(neg_items), p=2, dim=-1)
            neg_emb = neg_emb.view(interaction.length, self.max_seq_length * n_neg, -1)

            pos_emb = F.normalize(self.get_fused_target_embedding(pos_items), p=2, dim=-1)

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
        all_emb = F.normalize(self.get_fused_target_embedding(all_items), p=2, dim=-1)
        logits = torch.matmul(seq_output, all_emb.transpose(1, 0))
        logits *= self.logit_scale.exp().clamp(max=100)
        logits += self.item_bias()

        labels = torch.zeros_like(logits).scatter_add(
            dim=1, index=pos_items, src=torch.ones_like(pos_items).float()
        )
        labels[:, 0] = 0
        loss = sigmoid_focal_loss(logits, labels, reduction='none')

        return loss.mean()

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len, time_seq)

        test_item_emb = F.normalize(self.get_fused_target_embedding(test_item), p=2, dim=-1)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        seq_output = self.forward(item_seq, item_seq_len, time_seq)

        all_items = torch.arange(self.n_items, device=self.device)
        test_items_emb = F.normalize(self.get_fused_target_embedding(all_items), p=2, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores