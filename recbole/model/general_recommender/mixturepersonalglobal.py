r"""
MixturePersonalGlobal
################################################

"""

import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class MixturePersonalGlobal(GeneralRecommender):
    r"""MixturePersonalGlobal is a simple baseline model that recommends a linear combination of the most popular item
    and a user personal item preference."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(MixturePersonalGlobal, self).__init__(config, dataset)
        self.alpha = config["alpha"]
        self.user_purchase_per_item = torch.zeros(
            (self.n_items, self.n_users), device=self.device, requires_grad=False
        )
        self.user_purchase_count = torch.zeros(
            self.n_users, dtype=torch.float64, device=self.device, requires_grad=False
        )
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
        self.user_purchase_per_item[items, users] += 1
        self.item_purchase_count[items] += 1
        self.user_purchase_count[users] += 1
        self.total_purchase_count += interaction.length
        return self.fake_loss

    def predict(self, interaction):
        items, users = interaction[self.ITEM_ID], interaction[self.USER_ID]
        global_result = self.item_purchase_count[items].div(self.total_purchase_count)
        result = self.user_purchase_per_item[items, users].view(-1)
        result /= self.user_purchase_count[users]
        result = self.alpha * result + (1 - self.alpha) * global_result
        return result

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        global_result = self.item_purchase_count.div(self.total_purchase_count)
        result = self.user_purchase_per_item[:, users].view(-1)
        result /= self.user_purchase_count[users]
        result = self.alpha * result + (1 - self.alpha) * global_result
        return result.view(-1)


class MixtureGPGlobal(MixturePersonalGlobal):
    r"""MixtureGPGlobal is a simple baseline model that recommends a linear combination of the most popular item
    and a user personal item preference, the personal item preference vector are followed by the global popularity in
    case of items with no probabilities."""

    def predict(self, interaction):
        items, users = interaction[self.ITEM_ID], interaction[self.USER_ID]
        global_result = self.item_purchase_count[items].div(self.total_purchase_count)
        result = self.user_purchase_per_item[items, users].view(-1)

        # Filling empty slots in the personal vector with global popularitis
        result[torch.logical_not(result)] += global_result[torch.logical_not(result)]

        result /= (
            self.user_purchase_count[users] + 1
        )  # +1 here to take into accont the amount added by global result
        result = self.alpha * result + (1 - self.alpha) * global_result
        return result

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        global_result = self.item_purchase_count.div(self.total_purchase_count)
        result = self.user_purchase_per_item[:, users].view(-1)

        # Filling empty slots in the personal vector with global popularitis
        result[torch.logical_not(result)] += global_result[torch.logical_not(result)]

        result /= self.user_purchase_count[users] + 1
        result = self.alpha * result + (1 - self.alpha) * global_result
        return result.view(-1)
