r"""
TimeTopShopper
################################################

"""

import torch

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType, ModelType


class TimeTopShopper(SequentialRecommender):
    r"""PersonalPop is a fundamental model that always recommend a user's most purchased item"""

    input_type = InputType.POINTWISE
    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(TimeTopShopper, self).__init__(config, dataset)
        self.law = config["law"]
        assert self.law in ["hyperbolic", "exponential"]
        # self.parameter = config["parameter"]
        self.parameter = torch.nn.Linear(1)

    def forward(self, item_seq, time_seq):
        pass

    def calculate_loss(self, interaction):
        return self.fake_loss

    def predict(self, interaction):
        result = (
            (interaction[self.ITEM_SEQ] == interaction[self.ITEM_ID]).sum(dim=1).float()
        )
        return result

    def full_sort_predict(self, interaction):
        history = interaction[self.ITEM_SEQ]
        batch_size = interaction.length
        result = torch.zeros(
            (batch_size, self.n_items), device=self.device, dtype=torch.float
        )
        result = result.scatter_add(dim=1, src=(history > 0).float(), index=history)
        return result
