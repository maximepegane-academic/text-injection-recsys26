# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE:
# @Time   : 2022/7/8, 2020/9/16, 2021/7/1, 2021/7/11
# @Author : Zhen Tian, Yushuo Chen, Xingyu Pan, Yupeng Hou
# @Email  : chenyuwuxinn@gmail.com, chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, houyupeng@ruc.edu.cn

"""
recbole.data.sequential_dataset
###############################
"""

import numpy as np
import torch
import polars as pl
import pyarrow.parquet as pq
import gc
import tempfile
import os

from tqdm import tqdm

from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType, FeatureSource


class SequentialDataset(Dataset):
    """:class:`SequentialDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and provides augmentation interface to adapt to Sequential Recommendation,
    which can accelerate the data loader.

    Attributes:
        max_item_list_len (int): Max length of historical item list.
        item_list_length_field (str): Field name for item lists' length.
    """

    def __init__(self, config):
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        self.list_suffix = config["LIST_SUFFIX"]

        super().__init__(config)
        if config["benchmark_filename"] is not None:
            self._benchmark_presets()

        if self.config["optimize"]:
            self.purchase_counts = torch.tensor(
                self.inter_feat.group_by(self.uid_field)
                .agg(pl.col(self.time_field).max())
                .sort(by=self.uid_field)[self.time_field]
                .to_list(),
                device=config["device"],
            )

        else:
            self.purchase_counts = torch.tensor(
                self.inter_feat.groupby(self.uid_field)[self.time_field].max().tolist(),
                device=config["device"],
            )

    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """
        if self.config["optimize"]:
            if self.benchmark_filename_list is not None:
                part_ids = np.repeat(np.arange(len(self.file_size_list)), self.file_size_list)
                self.inter_feat = self.inter_feat.with_columns(pl.Series("_benchmark_part", part_ids))
                augmented_df = self.optimized_augmentation(self.inter_feat)

                self.inter_feat = augmented_df
            else:
                self.inter_feat = self.optimized_augmentation(self.inter_feat)
            self.field2type[self.item_list_length_field] = FeatureType.FLOAT
            self.field2type[self.iid_field] = FeatureType.TOKEN_SEQ
            self.field2type[self.iid_field + self.list_suffix] = FeatureType.TOKEN_SEQ
            self.field2type[self.time_field + self.list_suffix] = FeatureType.FLOAT_SEQ
            if not self.config["batch_data_processing"]:
                self.dense_feat_format()

        else:
            super()._change_feat_format()
            self.logger.debug("Augmentation for sequential recommendation.")
            self.data_augmentation()

    # CHANGED added a timestamp filtering, which effectively filters out user with an amount of timestamp (basket) outside
    # the defined interval in a similar fashion as _filter_inter_by_user_or_item in the base Dataset Object
    def _filter_by_num_timestamp(self):
        if self.config["user_amount_timestamp_num_interval"] is None:
            return
        if self.config["optimize"]:
            conf_str = self.config["user_amount_timestamp_num_interval"]
            lower_bound, upper_bound = map(
                float,
                self.config["user_amount_timestamp_num_interval"]
                .strip("[]()")
                .split(","),
            )
            lower_bound = max(lower_bound, 0.0)
            closed_map = {
                ("[", "]"): "both",
                ("[", ")"): "left",
                ("(", "]"): "right",
                ("(", ")"): "none",
            }
            strictness = closed_map.get((conf_str[0], conf_str[-1]), "both")
            self.inter_feat = self.inter_feat.filter(
                pl.col(self.time_field)
                .n_unique()
                .over(self.uid_field)
                .is_between(lower_bound, upper_bound, closed=strictness)
            )
        else:
            lower_bound, upper_bound = (
                self.config["user_amount_timestamp_num_interval"]
                .strip("[]()")
                .split(",")
            )
            lower_bound = max(float(lower_bound), 0)
            upper_bound = float(upper_bound)  # float incase it's inf
            num_basket_per_inter = self.inter_feat.groupby(self.uid_field)[
                self.time_field
            ].transform("max")

            self.inter_feat = self.inter_feat[
                (num_basket_per_inter >= lower_bound)
                & (num_basket_per_inter <= upper_bound)
            ]

    def _data_filtering(self):
        """Data filtering
        same as parent and comprises:
        - Filter missing user_id or item_id
        - Remove duplicated user-item interaction
        - Value-based data filtering
        - Remove interaction by user or item
        - Filter out users randomly to obtain a set amount of user
        - K-core data filtering # deprecated in this fork, only 1 pass filtering is applied

        with the addition of filtering out user based and on their number of basket
        """
        self._filter_nan_user_or_item()
        self._remove_duplication()
        self._filter_by_field_value()
        self._filter_inter_by_user_or_item()
        self._filter_by_inter_num()
        self._filter_by_num_timestamp()
        self._sample_uid_randomly()
        if not self.config["optimize"]:
            self._reset_index()

    def _aug_presets(self):
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + self.list_suffix
                setattr(self, f"{field}_list_field", list_field)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                self.set_field_property(
                    list_field, list_ftype, FeatureSource.INTERACTION, list_len
                )

        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )

    @staticmethod
    def build_item_id_lists(uid_inter_list, time_inter_list, max_item_list_len):
        last_uid = None
        last_timestamp = None
        item_list_index, target_index, item_list_length = [], [], []
        seq_start, seq_end = 0, 0
        for i, (uid, timestamp) in enumerate(zip(uid_inter_list, time_inter_list)):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
                corrected_start = i
                seq_end = i
                last_timestamp = timestamp

            else:
                if last_timestamp != timestamp:
                    last_timestamp = timestamp
                    seq_end = i

                if seq_end - seq_start > max_item_list_len:
                    corrected_start = seq_end - max_item_list_len
                else:
                    corrected_start = seq_start
            item_list_index.append(slice(corrected_start, seq_end))
            target_index.append(i)
            item_list_length.append(max(seq_end - corrected_start, 0))

        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)
        return item_list_index, target_index, item_list_length

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>`` for timestamp ``<t1, t1, t2, t3>``.
        then after augmentation, we will generate four cases.

        ``u1, <> | i1``

        ``u1, <> | i2``

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``

        Which means that given the sequence in the bracket <> for that user it will need to predict the item after `|`
        For each interaction, the model will have to predict it given a sequence from previous timestamps
        In the context of next basket prediction, predicting an interaction given the sequence of previous baskets

        The length of the <> sequence is determined by the `MAX_ITEM_LIST_LENGTH` config parameter.
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)

        item_list_index, target_index, item_list_length = self.build_item_id_lists(
            self.inter_feat[self.uid_field].cpu().numpy(),
            self.inter_feat[self.time_field].cpu().numpy(),
            max_item_list_len,
        )
        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                new_dict[list_field] = torch.zeros(
                    shape, dtype=self.inter_feat[field].dtype
                )

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    if length > 0:
                        new_dict[list_field][i][-length:] = value[index]
        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data

    def _benchmark_presets(self):
        feat_names = self.inter_feat.columns

        for field in feat_names:
            if field + self.list_suffix in feat_names:
                list_field = field + self.list_suffix
                setattr(self, f"{field}_list_field", list_field)
        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )

    def inter_matrix(self, form="coo", value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Sparse matrix has shape (user_num, item_num).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError(
                "dataset does not exist uid/iid, thus can not converted to sparse matrix."
            )

        l1_idx = self.inter_feat[self.item_list_length_field] == 1
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + self.list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat(
                    [
                        self.inter_feat[field],
                        l1_inter_dict[field + self.list_suffix][:, 0],
                    ]
                )
            elif (not field.endswith(self.list_suffix)) and (
                field != self.item_list_length_field
            ):
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field]]
                )
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(
            local_inter_feat, self.uid_field, self.iid_field, form, value_field
        )

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of built :class:`Dataset`.
        """
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )

        return super().build()

    def optimized_augmentation(self, dataset):
        mode = self.config.get("batch_data_processing", "auto")

        if mode is None or mode == "auto":
            mode = self.inter_feat.estimated_size() > 1024**3  # One GiB
        self.config["batch_data_processing"] = mode
        if mode:
            return self._process_by_user_chunks(dataset)
        return self._process_full(dataset)

    def _process_full(self, dataset):
        processed = self._apply_core_logic(dataset.lazy())

        del dataset
        gc.collect()

        return processed.lazy().collect(engine="streaming")

    def _process_by_user_chunks(self, dataset, users_per_chunk=50000):
        fd, output_path = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)

        lf = dataset.lazy()
        unique_users = (
            lf.select(self.uid_field).unique().collect().get_column(self.uid_field)
        )

        writer = None

        for i in tqdm(range(0, len(unique_users), users_per_chunk)):
            user_chunk = unique_users[i : i + users_per_chunk]
            chunk_lf = lf.filter(pl.col(self.uid_field).is_in(user_chunk))

            processed_chunk_df = self._apply_core_logic(chunk_lf)
            table = processed_chunk_df.to_arrow()

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)

            del processed_chunk_df
            del table
            gc.collect()

        if writer:
            writer.close()

        del dataset
        gc.collect()

        final_df = pl.scan_parquet(output_path)
        return final_df

    def _apply_core_logic(self, chunk_lf):
        df = chunk_lf.cast({self.time_field: pl.Int64, self.iid_field: pl.Int64})

        time_list_field = self.time_field + self.list_suffix
        iid_list_field = self.iid_field + self.list_suffix
        grouping = [self.uid_field, self.time_field]
        if self.benchmark_filename_list is not None:
            grouping.append("_benchmark_part")
        baskets = (
            df.group_by(grouping)
            .agg(
                [
                    pl.col(self.iid_field).alias("basket_items"),
                    pl.col(self.time_field).alias("basket_times"),
                ]
            )
            .sort([self.uid_field, self.time_field])
        )

        history = baskets.rolling(
            index_column=self.time_field,
            period=f"{self.max_item_list_len}i",
            group_by=self.uid_field,
            closed="left",
        ).agg(
            [
                pl.col("basket_items")
                .list.explode()
                .drop_nulls()
                .alias(iid_list_field),
                pl.col("basket_times")
                .list.explode()
                .drop_nulls()
                .alias(time_list_field),
            ]
        )

        result = (
            baskets.join(history, on=[self.uid_field, self.time_field], how="left")
            .with_columns(
                [
                    pl.col(iid_list_field).fill_null(
                        pl.lit([], dtype=pl.List(pl.Int64))
                    ),
                    pl.col(time_list_field).fill_null(
                        pl.lit([], dtype=pl.List(pl.Int64))
                    ),
                ]
            )
            .with_columns(
                pl.col(iid_list_field)
                .list.len()
                .cast(pl.Int64)
                .alias(self.item_list_length_field)
            )
        )

        return result.with_columns(
            [
                self.pad_sequence("basket_items", self.iid_field),
                self.pad_sequence(time_list_field, time_list_field),
                self.pad_sequence(iid_list_field, iid_list_field),
            ]
        ).select(
            pl.col("*").exclude("basket_items", "basket_times")
        ).sort(pl.int_range(pl.len()).shuffle()).collect()

    def pad_sequence(self, col_name, target_name):
        truncated = pl.col(col_name).list.tail(self.max_item_list_len)
        pad_len = self.max_item_list_len - truncated.list.len()

        return (
            pl.when(pad_len > 0)
            .then(pl.lit(0, dtype=pl.Int64).repeat_by(pad_len).list.concat(truncated))
            .otherwise(truncated)
            .list.to_array(self.max_item_list_len)
            .alias(target_name)
        )

    def convert_to_interaction(self, feat):
        df = feat
        new_dict = {}
        columns = df.columns

        for col in columns:
            field_type = self.field2type[col]
            if field_type == FeatureType.TOKEN:
                new_dict[col] = torch.LongTensor(df[col].cast(pl.Int64).to_list())

            if field_type == FeatureType.TOKEN_SEQ:
                new_dict[col] = torch.LongTensor(
                    df[col].cast(pl.List(pl.Int64)).to_list()
                )

            elif field_type == FeatureType.FLOAT:
                new_dict[col] = torch.FloatTensor(df[col].cast(pl.Float64).to_list())

            elif field_type == FeatureType.FLOAT_SEQ:
                new_dict[col] = torch.FloatTensor(
                    df[col].cast(pl.List(pl.Float64)).to_list()
                )

        return Interaction(new_dict)

    def dense_feat_format(self):
        self.inter_feat = self.convert_to_interaction(self.inter_feat)
        if self.user_feat is not None:
            self.user_feat = self.convert_to_interaction(self.user_feat)
        if self.item_feat is not None:
            self.item_feat = self.convert_to_interaction(self.item_feat)
