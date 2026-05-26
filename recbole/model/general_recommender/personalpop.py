r"""
PersonalPop
################################################

"""

import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType
from recbole.utils.case_study import full_sort_topk


class PersonalPop(GeneralRecommender):
    r"""PersonalPop is a fundamental model that always recommend a user's most purchased item"""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(PersonalPop, self).__init__(config, dataset)

        self.user_purchase_per_item = torch.zeros(
            (self.n_items, self.n_users), device=self.device, requires_grad=False
        )
        self.user_purchase_count = torch.zeros(
            self.n_users, dtype=torch.float64, device=self.device, requires_grad=False
        )
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        items, users = interaction[self.ITEM_ID], interaction[self.USER_ID]
        self.user_purchase_per_item[items, users] += 1
        self.user_purchase_count[users] += 1

        return self.fake_loss

    def predict(self, interaction):
        users, items = interaction[self.USER_ID], interaction[self.ITEM_ID]
        result = self.user_purchase_per_item[items, users].view(-1)
        result /= self.user_purchase_count[users]
        return result

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        result = self.user_purchase_per_item[:, users].view(-1)
        result /= self.user_purchase_count[users]
        return result.view(-1)
