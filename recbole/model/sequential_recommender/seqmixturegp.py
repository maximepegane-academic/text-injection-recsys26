r"""
MixturePersonalGlobal
################################################

"""

import torch

from recbole.model.sequential_recommender import SeqPersonalPop
from recbole.utils import InputType, ModelType


class SeqMixtureGP(SeqPersonalPop):
    r"""MixturePersonalGlobal is a simple baseline model that recommends a linear combination of the most popular item
    and a user personal item preference."""

    input_type = InputType.POINTWISE
    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SeqMixtureGP, self).__init__(config, dataset)
        self.alpha = config["alpha"]
        self.item_purchase_count = torch.zeros(
            self.n_items, device=self.device, requires_grad=False
        )
        self.total_purchase_count = torch.zeros(
            1, dtype=torch.float64, device=self.device, requires_grad=False
        )

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        items, users = interaction[self.ITEM_ID], interaction[self.USER_ID]
        super(SeqMixtureGP, self).calculate_loss(interaction)

        self.item_purchase_count[items] += 1
        self.total_purchase_count += interaction.length
        return self.fake_loss

    def predict(self, interaction):
        items, users = interaction[self.ITEM_ID], interaction[self.USER_ID]
        global_result = self.item_purchase_count[items].div(self.total_purchase_count)
        result = self.user_purchase_per_item[users, items].view(-1)
        result /= self.user_purchase_count[users]
        result = self.alpha * result + (1 - self.alpha) * global_result
        return result

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        global_result = self.item_purchase_count / self.total_purchase_count
        result = self.user_purchase_per_item / self.user_purchase_count[
            users
        ].unsqueeze(-1)
        result = self.alpha * result + (1 - self.alpha) * global_result
        return result.view(-1)
