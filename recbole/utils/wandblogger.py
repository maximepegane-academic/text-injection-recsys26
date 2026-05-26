# -*- coding: utf-8 -*-
# @Time   : 2022/8/2
# @Author : Ayush Thakur
# @Email  : ayusht@wandb.com

r"""
recbole.utils.wandblogger
################################
"""


class WandbLogger(object):
    def __init__(self, config):
        self.config = config
        self.log_wandb = config["log_wandb"]
        self.setup()

    def setup(self):
        if self.log_wandb:
            try:
                import wandb

                self._wandb = wandb
            except ImportError:
                raise ImportError(
                    "To use the Weights and Biases Logger please install wandb."
                    "Run `pip install wandb` to install it."
                )

            if self._wandb.run is None:
                project_name = self.config.get("wandb_project", "RecBole-Default")
                run_name = self.config.get("wandb_run_name", "generated-run")

                self._wandb.init(
                    project=project_name,
                    config=self.config,
                    name=run_name,
                )

            self._set_steps()

    def log_metrics(self, metrics, head="train", commit=True):
        if self.log_wandb:
            if head:
                metrics = self._add_head_to_metrics(metrics, head)
                self._wandb.log(metrics, commit=commit)
            else:
                self._wandb.log(metrics, commit=commit)

    def log_eval_metrics(self, metrics, head="eval"):
        if self.log_wandb:
            metrics = self._add_head_to_metrics(metrics, head)
            for k, v in metrics.items():
                self._wandb.run.summary[k] = v

    def _set_steps(self):
        self._wandb.define_metric("train/*", step_metric="train_step")
        self._wandb.define_metric("valid/*", step_metric="valid_step")

    def _add_head_to_metrics(self, metrics, head):
        head_metrics = dict()
        for k, v in metrics.items():
            if "_step" in k:
                head_metrics[k] = v
            else:
                head_metrics[f"{head}/{k}"] = v

        return head_metrics
