import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.loss import sigmoid_focal_loss
from transformers import AutoConfig, AutoModel, AutoTokenizer
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import  ItemBias
from recbole.utils import InputType, build_text_col
from tqdm import tqdm
import math
import numpy as np

from recbole.utils.utils import batched_isin

class CAT(SequentialRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        build_text_col(config, dataset)
        self.hf_name = config["hugging_face_model"]
        self.hidden_size = config["hidden_size"]

        self.n_heads = config["n_heads"]
        self.inner_size = self.hidden_size * 4
        self.max_seq_len = config["MAX_ITEM_LIST_LENGTH"]
        self.n_layers = config["n_layers"]
        self.text_field = dataset.text_field
        self.full_nlp = config["full_nlp"]
        self.repeat_loss_scale = config["repeat_loss_scale"]
        self.dropout = config["dropout"]
        self.emb_dropout = nn.Dropout(config["dropout"])

        self.scale_factor = math.sqrt(self.hidden_size)

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_len + 1, self.hidden_size, padding_idx=0)
        self.text_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.gate_temp = nn.Parameter(torch.tensor(2.3))
        self.gate_layer = nn.Linear(self.hidden_size*2, 1, bias=False)
        self.nlp_proj = None
        if self.hf_name:
            hf_config = AutoConfig.from_pretrained(self.hf_name)
            self.hf_hidden_size = hf_config.hidden_size
            self.nlp_proj = nn.Linear(self.hf_hidden_size, self.hidden_size, bias=False)

        if self.train:
            self.build_nlp_lookup_table(dataset)

        self.text_embedding.weight.requires_grad = False

        cross_layer_block = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            batch_first=True,
            dropout=self.dropout
        )
        self.cross_layer = nn.TransformerDecoder(cross_layer_block, num_layers=1)
        self.sequence_encoder = None
        if self.n_layers>1:
            deep_layer_block = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.n_heads,
                dim_feedforward=self.inner_size,
                batch_first=True,
                dropout=self.dropout
            )
            self.sequence_encoder = nn.TransformerEncoder(deep_layer_block, num_layers=self.n_layers - 1)


        self.item_bias = ItemBias(dataset)
        print(
            f"CHECK: item_num is {self.n_items}, text_embedding size is {self.text_embedding.weight.shape[0]}")
        if config["optimize"]:
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.n_items / 8], device=self.device))
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if module is self.text_embedding:
            return
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
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

        if not self.hf_name:
            return

        # 1. Handle Pre-cached NLP Embeddings (full_nlp mode)
        if self.full_nlp:
            cache_path = dataset.config.get("nlp_cache_path", "nlp_cache.pt")
            print(f"Loading embeddings from NLP cache: {cache_path}")

            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            catalog_embs = cache["embs"]
            catalog_offsets = cache["offsets"]

            self.text_embedding = nn.Embedding(self.n_items, self.hf_hidden_size, padding_idx=0, device=self.device)
            cls_embs = catalog_embs[catalog_offsets[:-1]]
            cls_embs = F.normalize(cls_embs.float(), p=2, dim=-1)

            self.text_embedding.weight.data.copy_(cls_embs)
            del cache, catalog_embs, catalog_offsets
            return

        print(f"Precomputing NLP Embeddings from {self.hf_name}...")
        tokenizer = AutoTokenizer.from_pretrained(self.hf_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        encoder = AutoModel.from_pretrained(self.hf_name).to(self.device)
        encoder.eval()

        hf_lower = self.hf_name.lower()
        if any(kw in hf_lower for kw in ["llama", "mistral", "gemma", "opt", "gpt", "qwen"]):
            default_strategy = "last_token"
            default_prompt = "Represent the product description: "
        elif any(kw in hf_lower for kw in ["minilm", "e5", "sentence-transformers", "bge", "all-mpnet"]):
            default_strategy = "mean"
            default_prompt = "passage: " if "e5" in hf_lower else ""
        else:
            default_strategy = "cls"
            default_prompt = ""

        pooling_strategy = dataset.config.get("nlp_pooling_strategy", default_strategy)
        prompt_prefix = dataset.config.get("nlp_prompt_prefix", default_prompt)

        print(f"Extraction Strategy: {pooling_strategy.upper()} | Prompt: '{prompt_prefix}'")

        if self.text_field and self.text_field in dataset.item_feat.columns:
            item_texts = dataset.item_feat.sort(self.ITEM_ID)[self.text_field].to_list()
        else:
            item_texts = [""] * self.n_items

        item_texts[0] = ""

        batch_size = dataset.config.get("nlp_batch_size", 128)
        max_len = dataset.config.get("nlp_max_length", 128)
        embs = []

        for i in tqdm(range(0, self.n_items, batch_size), desc="Encoding NLP"):
            batch_texts = item_texts[i: i + batch_size]

            if prompt_prefix:
                batch_texts = [prompt_prefix + text if text.strip() else "" for text in batch_texts]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(self.device)

            out = encoder(**inputs).last_hidden_state

            if pooling_strategy == "mean":
                attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
                sum_embeddings = torch.sum(out * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                pooled_emb = sum_embeddings / sum_mask

            elif pooling_strategy == "last_token":
                sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1
                batch_indices = torch.arange(out.size(0), device=self.device)
                pooled_emb = out[batch_indices, sequence_lengths, :]

            elif pooling_strategy == "cls":
                pooled_emb = out[:, 0, :]

            else:
                raise ValueError(f"Unknown nlp_pooling_strategy: {pooling_strategy}")

            embs.append(pooled_emb.cpu())

        self.text_embedding = nn.Embedding(self.n_items, self.hf_hidden_size, padding_idx=0, device=self.device)
        self.text_embedding.weight.data.copy_(torch.cat(embs, dim=0))

        del encoder
        torch.cuda.empty_cache()

    def make_basket_causal_mask(self, basket_indices):
        query_time = basket_indices.unsqueeze(2)
        key_time = basket_indices.unsqueeze(1)
        causal_mask = key_time > query_time
        return causal_mask.repeat_interleave(self.n_heads, dim=0)

    def forward(self, interaction):
        item_seq, time_seq = interaction[self.ITEM_SEQ], interaction[self.TIME_SEQ]
        basket_indices = time_seq.long()
        min_time = (basket_indices.min(dim=1).values - 1)
        basket_indices = (basket_indices - min_time.unsqueeze(1)).clamp(0, self.max_seq_len)

        id_emb = self.item_embedding(item_seq) + self.position_embedding(basket_indices)
        id_emb = self.emb_dropout(id_emb)

        tgt_causal_mask = self.make_basket_causal_mask(basket_indices)
        pad_mask = (item_seq == 0)

        if self.full_nlp:
            text_emb = interaction["nlp_embeddings"]
            causal_mask = interaction["nlp_cross_causal_mask"].repeat_interleave(self.n_heads, dim=0)
            memory_pad_mask = ~(interaction["nlp_attention_mask"].bool())
        else:
            text_emb = self.text_embedding(item_seq)
            causal_mask = tgt_causal_mask
            memory_pad_mask = pad_mask
        if self.nlp_proj is not None:
            text_emb = self.nlp_proj(text_emb)

        seq_output = self.cross_layer(
            tgt=id_emb,
            memory=text_emb,
            tgt_mask=tgt_causal_mask,
            memory_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )

        if self.sequence_encoder is not None:
            seq_output = self.sequence_encoder(
                src=seq_output,
                mask=tgt_causal_mask,
                src_key_padding_mask=pad_mask
            )

        return F.normalize(seq_output[:, -1, :], p=2, dim=-1)

    def get_fused_target_embedding(self, item_ids):
        id_emb = self.item_embedding(item_ids)
        text_emb = self.text_embedding(item_ids)

        if self.nlp_proj is not None:
            text_emb = self.nlp_proj(text_emb)
        id_emb = F.normalize(id_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        cat_emb = torch.cat([id_emb, text_emb.detach()], dim=-1)
        logits = self.gate_layer(cat_emb)
        tau = self.gate_temp
        modality_weight = torch.sigmoid(logits / tau)

        fused_emb = (modality_weight * text_emb) + ((1.0 - modality_weight) * id_emb)
        return F.normalize(fused_emb, p=2, dim=-1)

    def get_info_nce_loss(self, item_ids):
        id_emb = self.item_embedding(item_ids)
        text_emb = self.text_embedding(item_ids)
        if self.nlp_proj is not None:
            text_emb = self.nlp_proj(text_emb)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        id_emb = F.normalize(id_emb, p=2, dim=1)

        logits = text_emb @ id_emb.T

    def calculate_loss(self, interaction):
        seq_output = self.forward(interaction)
        pos_items = interaction[self.ITEM_ID]
        if self.NEG_ITEM_ID in interaction:
            neg_items = interaction[self.NEG_ITEM_ID]
            n_neg = neg_items.shape[2]
            neg_items = neg_items.view(interaction.length, self.max_seq_len * n_neg)
            neg_emb = self.get_fused_target_embedding(neg_items)# F.normalize(self.item_embedding(neg_items), p=2, dim=-1)
            neg_emb = neg_emb.view(interaction.length, self.max_seq_len * n_neg, -1)

            pos_emb = self.get_fused_target_embedding(pos_items) # F.normalize(self.item_embedding(pos_items), p=2, dim=-1)

            pos_bias = self.item_bias(pos_items) #+ self.user_bias(interaction[self.ITEM_SEQ], pos_items)
            neg_bias = self.item_bias(neg_items).view(interaction.length, self.max_seq_len * n_neg) # + self.user_bias(

            pos_logits = (seq_output.unsqueeze(1) * pos_emb).sum(-1) * self.logit_scale.exp() + pos_bias
            neg_logits = (seq_output.unsqueeze(1) * neg_emb).sum(-1) * self.logit_scale.exp() + neg_bias
            logits = torch.cat([pos_logits, neg_logits], dim=-1)

            labels = torch.tensor([1.0] * self.max_seq_len + [0.0] * n_neg * self.max_seq_len,
                                  device=self.device).expand_as(logits)

            raw_loss = sigmoid_focal_loss(logits, labels, reduction='none')
            valid_mask = (pos_items > 0).repeat(1, n_neg + 1)
            masked_loss = raw_loss * valid_mask.float()

            return masked_loss.sum() / (valid_mask.sum() * 2 + 1e-9)

        all_items = torch.arange(self.n_items, device=self.device)
        all_fused_emb = self.get_fused_target_embedding(all_items)
        logits = torch.matmul(seq_output, all_fused_emb.transpose(1, 0))
        logits *= self.logit_scale.clamp(max=100)
        logits += self.item_bias()

        labels = torch.zeros_like(logits).scatter_add(
            dim=1, index=pos_items, src=torch.ones_like(pos_items).float()
        )
        labels[:, 0] = 0
        raw_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
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

    def full_sort_predict(self, interaction):
        seq_output = self.forward(interaction)

        all_items = torch.arange(self.n_items, device=self.device)
        all_fused_emb = self.get_fused_target_embedding(all_items)

        return torch.matmul(seq_output, all_fused_emb.transpose(0, 1))

    def predict(self, interaction):
        pass