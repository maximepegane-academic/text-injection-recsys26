import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import dok_matrix
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import ItemBias
from recbole.model.loss import sigmoid_focal_loss
from recbole.utils import InputType, build_text_col
from recbole.utils.utils import batched_isin


class GatedPoolingRecommender(SequentialRecommender):
    """
    A Late-Fusion Multimodal Recommender. 
    It mean-pools the user's behavioral and semantic history into a profile vector, 
    then uses a dynamic neural gate to blend collaborative and NLP similarities.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        build_text_col(config, dataset)

        # Configuration
        self.hf_name = config["hugging_face_model"]
        self.hidden_size = config["hidden_size"]
        self.max_seq_len = config["MAX_ITEM_LIST_LENGTH"]
        self.text_field = dataset.text_field
        self.repeat_loss_scale = config["repeat_loss_scale"]
        self.id_init_method = config.get("id_init_method", "random")

        # Modality Encoders
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0, device=self.device)
        self.text_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0, device=self.device)
        self.item_bias = ItemBias(dataset)

        # Gating Mechanism
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.gate_temp = nn.Parameter(torch.tensor(0.5))
        self.gate_layer = nn.Linear(self.hidden_size * 2, 1, bias=False, device=self.device)

        # NLP Projection
        self.nlp_proj = None
        if self.hf_name:
            hf_config = AutoConfig.from_pretrained(self.hf_name)
            self.hf_hidden_size = hf_config.hidden_size
            self.nlp_proj = nn.Linear(self.hf_hidden_size, self.hidden_size, bias=False, device=self.device)

        # Initialization
        if self.train:
            self.build_nlp_lookup_table(dataset)

        self.text_embedding.weight.requires_grad = False
        self.apply(self._init_weights)

        if self.id_init_method in ["cocount_pca", "prod2vec"]:
            self._initialize_id_embedding(dataset)

        # Loss Function
        if config["optimize"]:
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.n_items / 8], device=self.device))
        else:
            self.loss_fct = nn.CrossEntropyLoss()

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
        if dataset.config.get("inference", False) or not self.hf_name:
            self.text_embedding = nn.Embedding(self.n_items, self.hf_hidden_size, padding_idx=0, device=self.device)
            return

        tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
        encoder = AutoModel.from_pretrained(self.hf_name).to(self.device)
        encoder.eval()

        if self.text_field and self.text_field in dataset.item_feat.columns:
            item_texts = dataset.item_feat.sort(self.ITEM_ID)[self.text_field].to_list()
        else:
            item_texts = [""] * self.n_items
        item_texts[0] = ""

        batch_size = 128
        embs = []

        for i in tqdm(range(0, self.n_items, batch_size), desc="Encoding NLP"):
            batch_texts = item_texts[i: i + batch_size]
            if "e5" in self.hf_name.lower():
                batch_texts = ["passage: " + text for text in batch_texts]

            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(
                self.device)
            out = encoder(**inputs).last_hidden_state

            if "e5" in self.hf_name.lower() or "minilm" in self.hf_name.lower():
                attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
                sum_embeddings = torch.sum(out * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                pooled_emb = sum_embeddings / sum_mask
            else:
                pooled_emb = out[:, 0, :]

            embs.append(pooled_emb.cpu())

        self.text_embedding = nn.Embedding(self.n_items, self.hf_hidden_size, padding_idx=0, device=self.device)
        self.text_embedding.weight.data.copy_(torch.cat(embs, dim=0))

        del encoder
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _initialize_id_embedding(self, dataset):
        user_ids = dataset.inter_feat.select(self.USER_ID).lazy().collect().to_series().to_numpy()
        item_ids = dataset.inter_feat.select(self.ITEM_ID).lazy().collect().to_series().to_numpy()

        seqs = {}
        for u, i in zip(user_ids, item_ids):
            if u not in seqs:
                seqs[u] = []
            seqs[u].append(i)
        sentences = list(seqs.values())

        if self.id_init_method == "prod2vec":
            sentences_str = [[str(i) for i in seq] for seq in sentences]
            w2v = Word2Vec(sentences_str, vector_size=self.hidden_size, window=5, min_count=1, workers=4)
            for i in range(1, self.n_items):
                if str(i) in w2v.wv:
                    self.item_embedding.weight.data[i] = torch.tensor(w2v.wv[str(i)], device=self.device)

        elif self.id_init_method == "cocount_pca":
            co_mat = dok_matrix((self.n_items, self.n_items), dtype=np.float32)
            for seq in sentences:
                for i in range(len(seq)):
                    for j in range(max(0, i - 5), min(len(seq), i + 6)):
                        if i != j:
                            co_mat[seq[i], seq[j]] += 1
            co_mat = co_mat.tocsr()
            svd = TruncatedSVD(n_components=self.hidden_size, random_state=42)
            embs = svd.fit_transform(co_mat)
            self.item_embedding.weight.data[1:] = torch.tensor(embs[1:], dtype=torch.float32, device=self.device)

    def get_item_representations(self, item_ids):
        """Fetches, normalizes, and computes the dynamic gate for target items."""
        id_emb = F.normalize(self.item_embedding(item_ids), p=2, dim=-1)
        text_emb = self.text_embedding(item_ids)

        if self.nlp_proj is not None:
            text_emb = self.nlp_proj(text_emb)

        text_emb = F.normalize(text_emb, p=2, dim=-1)
        cat_emb = torch.cat([id_emb, text_emb.detach()], dim=-1)

        tau = F.softplus(self.gate_temp) + 1e-4
        alpha = torch.sigmoid(self.gate_layer(cat_emb) / tau)

        if alpha.shape[-1] == 1:
            alpha = alpha.squeeze(-1)

        return id_emb, text_emb, alpha

    def forward(self, item_seq):
        """
        Creates the User Profile. 
        Averages the user's historical items into single behavioral and semantic vectors.
        """
        hist_id, hist_nlp, _ = self.get_item_representations(item_seq)

        mask = (item_seq > 0).float()
        denom = mask.sum(dim=1, keepdim=True) + 1e-9

        user_id_profile = (hist_id * mask.unsqueeze(-1)).sum(dim=1) / denom
        user_nlp_profile = (hist_nlp * mask.unsqueeze(-1)).sum(dim=1) / denom

        return user_id_profile, user_nlp_profile

    def calculate_scores(self, item_seq, target_items):
        """Calculates gated cosine similarity between user profile and target items."""
        user_id_profile, user_nlp_profile = self.forward(item_seq)
        targ_id, targ_nlp, targ_alpha = self.get_item_representations(target_items)

        if target_items.dim() == 1:
            # Eval Mode: Vector x Matrix Math
            sim_id = torch.matmul(user_id_profile, targ_id.t())
            sim_nlp = torch.matmul(user_nlp_profile, targ_nlp.t())
            targ_alpha = targ_alpha.view(1, -1)
        else:
            # Train Mode: Batch Negative Sampling Math
            sim_id = (user_id_profile.unsqueeze(1) * targ_id).sum(dim=-1)
            sim_nlp = (user_nlp_profile.unsqueeze(1) * targ_nlp).sum(dim=-1)

        # Apply Neural Gating In-Place (Memory efficient formulation)
        sim_nlp.sub_(sim_id)
        sim_nlp.mul_(targ_alpha)
        sim_id.add_(sim_nlp)

        return sim_id

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        pos_items = interaction[self.ITEM_ID]

        # Pairwise Loss
        if self.NEG_ITEM_ID in interaction:
            neg_items = interaction[self.NEG_ITEM_ID]
            n_neg = neg_items.shape[2]
            neg_items = neg_items.view(interaction.length, self.max_seq_len * n_neg)

            pos_scores = self.calculate_scores(item_seq, pos_items)
            neg_scores = self.calculate_scores(item_seq, neg_items)

            pos_bias = self.item_bias(pos_items)
            if pos_bias.dim() == 3:
                pos_bias = pos_bias.squeeze(-1)

            neg_bias = self.item_bias(neg_items).view(interaction.length, self.max_seq_len * n_neg)

            pos_logits = pos_scores * self.logit_scale.exp() + pos_bias
            neg_logits = neg_scores * self.logit_scale.exp() + neg_bias
            logits = torch.cat([pos_logits, neg_logits], dim=-1)

            labels = torch.tensor([1.0] * self.max_seq_len + [0.0] * n_neg * self.max_seq_len,
                                  device=self.device).expand_as(logits)

            raw_loss = sigmoid_focal_loss(logits, labels, reduction='none')
            valid_mask = (pos_items > 0).repeat(1, n_neg + 1)
            masked_loss = raw_loss * valid_mask.float()

            return masked_loss.sum() / (valid_mask.sum() * 2 + 1e-9)

        # Pointwise Loss
        all_items = torch.arange(self.n_items, device=self.device)
        logits = self.calculate_scores(item_seq, all_items)
        logits *= self.logit_scale.clamp(max=100)
        logits += self.item_bias().squeeze(-1)

        labels = torch.zeros_like(logits).scatter_add(
            dim=1, index=pos_items, src=torch.ones_like(pos_items).float()
        )
        labels[:, 0] = 0
        raw_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        repeat_mask = batched_isin(pos_items, item_seq)

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
        item_seq = interaction[self.ITEM_SEQ]
        all_items = torch.arange(self.n_items, device=self.device)
        return self.calculate_scores(item_seq, all_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        target_item = interaction[self.ITEM_ID]
        return self.calculate_scores(item_seq, target_item)