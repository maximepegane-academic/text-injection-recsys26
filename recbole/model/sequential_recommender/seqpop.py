# -*- coding: utf-8 -*-
# @Time   : 2020/8/11 9:57
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com
# UPDATE
# @Time   : 2020/11/9
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn
# UPDATE
# @Time   :2023/9/21
# @Author : Kesha Ou
# @Email  :1582706091@qq.com

r"""
Pop
################################################

"""

import torch

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType, ModelType


class SeqPop(SequentialRecommender):
    r"""Pop is an fundamental model that always recommend the most popular item."""

    input_type = InputType.POINTWISE
    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SeqPop, self).__init__(config, dataset)

        self.pop = dataset.item_popularity
        self.pop = self.pop / self.pop.sum()
        self.pop = self.pop.to(self.device)

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def calculate_loss(self, interaction):
        return self.fake_loss.mean()

    def predict(self, interaction):
        batch_size = interaction.length
        result = self.pop[interaction[self.ITEM_ID]]
        return result.repeat(batch_size, 1)

    def full_sort_predict(self, interaction):
        batch_size = interaction.length
        result = torch.zeros(
            (batch_size, self.n_items), dtype=torch.float, device=self.device
        )
        result += self.pop
        return result
