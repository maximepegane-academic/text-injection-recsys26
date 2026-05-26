import os
import torch
import polars as pl
from recbole.data.interaction import Interaction
from recbole.utils import build_text_col
from transformers import AutoTokenizer, AutoModel
import math
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm


class IterTrainDataloader:
    def __init__(self, config, dataset, sampler, shuffle=False):
        self.config = config
        self._sampler = sampler
        self._dataset = dataset
        self.dataset = dataset

        self.batch_size = self.config["train_batch_size"]
        self.times = (
            max(self.config["train_neg_sample_args"]["sample_num"], 1)
            if self.config["train_neg_sample_args"]["distribution"] != "none"
            else 1
        )
        self.read_chunk_size = self.batch_size  # max(self.batch_size // self.times, 1)
        self.step_size = self.read_chunk_size
        self.total_rows = dataset.inter_feat.select(pl.len()).collect().item()
        self.total_batches = self.total_rows // self.step_size

        self.tokenizer = None
        self.item_texts = None
        self.hf_model_name = None

        if self.dataset.config.get("full_nlp", False):
            self.hf_model_name = self.dataset.config.get("hugging_face_model")

            # We don't need the HuggingFace tokenizer anymore during __iter__,
            # but we pass the model name down so StreamingDataset can use it for extraction.

            text_field = build_text_col(self.config, self.dataset)

            if text_field and text_field in self.dataset.item_feat.columns:
                text_df = self.dataset.item_feat.sort(self.dataset.iid_field)
                self.item_texts = text_df.get_column(text_field).to_list()
            else:
                self.item_texts = [""] * self.dataset.item_num

            if len(self.item_texts) > 0:
                self.item_texts[0] = ""

        self.iterable_dataset = StreamingDataset(
            dataset=self._dataset,
            sampler=sampler,
            batch_size=self.step_size,
            is_train=True,
            item_texts=self.item_texts,
            hf_model_name=self.hf_model_name
        )
        self._dataloader = DataLoader(self.iterable_dataset, batch_size=None, pin_memory=True, num_workers=0)

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        return self.total_batches

    def get_model(self, model):
        self.model = model


class StreamingDataset(IterableDataset):
    def __init__(self, dataset, sampler, batch_size, is_train, item_texts=None, hf_model_name=None,
                 precomputed_embs=None, precomputed_offsets=None):
        super().__init__()
        self._dataset = dataset
        self._sampler = sampler
        self.batch_size = batch_size
        self.max_seq_len = dataset.max_item_list_len
        self.is_train = is_train

        self.full_nlp = dataset.config.get("full_nlp", False)
        self.item_texts = item_texts
        self.hf_model_name = hf_model_name
        self.max_text_len = dataset.config.get("max_text_len", 512)
        self.cache_path = dataset.config.get("nlp_cache_path", "nlp_cache.pt")

        self.ITEM_ID = dataset.config["ITEM_ID_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + dataset.config["LIST_SUFFIX"]
        self.TIME_FIELD = dataset.config["TIME_FIELD"]
        self.TIME_SEQ = self.TIME_FIELD + dataset.config["LIST_SUFFIX"]

        if self.is_train:
            self.neg_sample_args = dataset.config["train_neg_sample_args"]
            self.neg_sample_num = self.neg_sample_args["sample_num"]
            self.times = self.neg_sample_num
            self.neg_prefix = dataset.config["NEG_PREFIX"]

        self.catalog_embs = precomputed_embs
        self.catalog_offsets = precomputed_offsets

        if self.full_nlp and self.catalog_embs is None:
            if os.path.exists(self.cache_path):
                print(f"[{'Train' if self.is_train else 'Eval'} Dataset] Loading NLP cache from {self.cache_path}...")
                cache = torch.load(os.path.abspath(self.cache_path), map_location="cpu", weights_only=True)
                self.catalog_embs = cache["embs"]
                self.catalog_offsets = cache["offsets"]
            else:
                print(
                    f"[{'Train' if self.is_train else 'Eval'} Dataset] No cache found. Computing embeddings natively...")
                self._precompute_catalog_embeddings()

    def _precompute_catalog_embeddings(self):
        """Extracts and saves the Hugging Face embeddings for the entire catalog."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        model = AutoModel.from_pretrained(self.hf_model_name).to(device)
        model.eval()

        all_embs = []
        offsets = [0]
        current_offset = 0
        batch_size = 128

        with torch.no_grad():
            for i in tqdm(range(0, len(self.item_texts), batch_size), desc="Extracting NLP Features"):
                batch_texts = self.item_texts[i: i + batch_size]

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(device)

                outputs = model(**inputs).last_hidden_state

                for b in range(len(batch_texts)):
                    seq_len = inputs["attention_mask"][b].sum().item()
                    valid_emb = outputs[b, :seq_len, :].cpu().half()

                    all_embs.append(valid_emb)
                    current_offset += seq_len
                    offsets.append(current_offset)

        master_tensor = torch.cat(all_embs, dim=0)
        offset_tensor = torch.tensor(offsets, dtype=torch.long)

        del model
        torch.cuda.empty_cache()

        torch.save({"embs": master_tensor, "offsets": offset_tensor}, self.cache_path)
        print("NLP Cache successfully saved to disk!")

        self.catalog_embs = master_tensor
        self.catalog_offsets = offset_tensor

    def __iter__(self):
        lf = self._dataset.inter_feat

        for batch_df in lf.collect_batches(chunk_size=self.batch_size):
            batch_dict = {}

            for col in batch_df.columns:
                batch_dict[col] = batch_df[col].to_torch().clone()

            if self.full_nlp:
                history_sequences = batch_dict[self.ITEM_SEQ]
                time_sequences = batch_dict[self.TIME_SEQ]
                hidden_size = self.catalog_embs.shape[1]

                B, S = history_sequences.shape
                valid_mask = history_sequences != 0

                valid_items = history_sequences[valid_mask]
                valid_times = time_sequences[valid_mask]
                batch_indices = \
                    torch.arange(B, device=history_sequences.device).unsqueeze(1).expand_as(history_sequences)[
                        valid_mask]

                starts = self.catalog_offsets[valid_items]
                ends = self.catalog_offsets[valid_items + 1]
                lengths = ends - starts

                total_tokens = lengths.sum().item()

                if total_tokens == 0:
                    batch_user_embs = [torch.empty((0, hidden_size), dtype=torch.float16,
                                                   device=history_sequences.device)] * B
                    batch_user_times = [torch.empty((0,), dtype=torch.long, device=history_sequences.device)] * B
                else:
                    base = torch.repeat_interleave(starts, lengths)
                    cum_lengths = F.pad(lengths.cumsum(0), (1, 0))
                    cum_repeated = torch.repeat_interleave(cum_lengths[:-1], lengths)
                    idx_offsets = torch.arange(total_tokens, device=history_sequences.device) - cum_repeated

                    flat_indices = base + idx_offsets

                    flat_embs = self.catalog_embs[flat_indices]
                    flat_times = torch.repeat_interleave(valid_times, lengths)

                    tokens_per_user = torch.bincount(batch_indices, weights=lengths.float(), minlength=B).long()

                    batch_user_embs = list(torch.split(flat_embs, tokens_per_user.tolist()))
                    batch_user_times = list(torch.split(flat_times, tokens_per_user.tolist()))

                batch_embs = []
                batch_token_times = []

                for emb, times in zip(batch_user_embs, batch_user_times):
                    if emb.shape[0] > 0:
                        cat_embs = emb[:self.max_text_len]
                        cat_times = times[:self.max_text_len]
                    else:
                        cat_embs = torch.zeros((1, hidden_size), dtype=torch.float16, device=history_sequences.device)
                        cat_times = torch.zeros((1,), dtype=torch.long, device=history_sequences.device)

                    batch_embs.append(cat_embs)
                    batch_token_times.append(cat_times)

                padded_embs = pad_sequence(batch_embs, batch_first=True, padding_value=0.0)
                padded_times = pad_sequence(batch_token_times, batch_first=True, padding_value=0)

                seq_lens = torch.tensor([x.shape[0] for x in batch_embs], device=padded_embs.device)
                attention_mask = torch.arange(padded_embs.shape[1], device=padded_embs.device)[None, :] < seq_lens[
                    :, None]

                query_timestamps = time_sequences.unsqueeze(2)
                key_timestamps = padded_times.unsqueeze(1)
                cross_causal_mask = key_timestamps > query_timestamps

                batch_dict["nlp_embeddings"] = padded_embs.float()
                batch_dict["nlp_attention_mask"] = attention_mask
                batch_dict["nlp_cross_causal_mask"] = cross_causal_mask

            interaction = Interaction(batch_dict)

            if self.is_train and self.neg_sample_args["distribution"] != "none":
                neg_item_ids = self._sampler.sample_by_interaction(interaction, self.neg_sample_num)

                # interaction = interaction.repeat(self.times)
                neg_item_feat = Interaction({self._dataset.iid_field: neg_item_ids})
                neg_item_feat.add_prefix(self.neg_prefix)
                interaction.update(neg_item_feat)

            yield interaction


class IterEvalDataloader(IterTrainDataloader):
    def __init__(self, config, dataset, sampler, shuffle=False, train_dataloader=None):
        super().__init__(config, dataset, sampler, shuffle)

        precomputed_embs = None
        precomputed_offsets = None

        # Memory-sharing optimization: If train dataloader already loaded the cache, reuse its tensors!
        if train_dataloader is not None and hasattr(train_dataloader, 'iterable_dataset'):
            precomputed_embs = train_dataloader.iterable_dataset.catalog_embs
            precomputed_offsets = train_dataloader.iterable_dataset.catalog_offsets

        self.batch_size = self.config["eval_batch_size"]
        self.step = self.batch_size
        self.total_rows = dataset.inter_feat.select(pl.len()).collect().item()
        self.total_batches = math.ceil(self.total_rows / self.batch_size)

        self.iterable_dataset = StreamingDataset(
            dataset=self._dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            is_train=False,
            item_texts=self.item_texts,
            hf_model_name=self.hf_model_name,
            precomputed_embs=precomputed_embs,
            precomputed_offsets=precomputed_offsets
        )
        self._dataloader = DataLoader(self.iterable_dataset, batch_size=None, num_workers=0)

    def __iter__(self):
        for interaction in self._dataloader:
            inter_num = len(interaction)
            positive_u = torch.arange(inter_num)
            positive_i = interaction[self._dataset.iid_field]

            yield interaction, None, positive_u, positive_i
