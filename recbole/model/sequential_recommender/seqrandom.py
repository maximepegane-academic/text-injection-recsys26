r"""
Random
################################################

"""

import torch

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType, ModelType


class SeqRandom(SequentialRecommender):
    r"""Pop is an fundamental model that always recommend the most popular item."""

    input_type = InputType.POINTWISE
    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SeqRandom, self).__init__(config, dataset)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return self.fake_loss

    def predict(self, interaction):
        return torch.rand(interaction.length)

    def full_sort_predict(self, interaction):
        n_lines, n_items = interaction.length, self.n_items
        result = torch.rand(n_lines * n_items)
        result = result.reshape(n_lines, n_items)
        return result.view(-1)
