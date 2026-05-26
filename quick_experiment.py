import sys
import gc
import logging
import torch.distributed as dist
from torch.cuda import empty_cache

import torch
torch.set_float32_matmul_precision('high')

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_environment,
)

import utils


def run_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    config, logger = _setup_environment(model, dataset, config_file_list, config_dict)

    dataset_obj = create_dataset(config)
    logger.info(dataset_obj)

    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    model_instance = _setup_model(config, train_data, dataset_obj, logger)

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model_instance)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    test_result = _run_evaluation(trainer, test_data, saved, config)

    _log_final_results(logger, best_valid_result, test_result, config)

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "config": config,
    }

    utils.save_experiment_results(model_instance, result, config, trainer)


    _cleanup(config, queue, result)

    return result


def _setup_environment(model, dataset, config_file_list, config_dict):
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])

    init_logger(config)
    logger = logging.getLogger()
    logger.info(sys.argv)
    logger.info(config)

    return config, logger


def run_inference(model_path, config_dict=None):
    """
    Loads a trained model and evaluates it directly on the test set.
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    if config_dict:
        config.internal_config_dict.update(config_dict)

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = logging.getLogger()
    logger.info(f"Loading model for inference from: {model_path}")
    logger.info(config)

    dataset_obj = create_dataset(config)
    logger.info(dataset_obj)

    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])

    if "other_parameter" in checkpoint:
        model.load_other_parameter(checkpoint["other_parameter"])

    model.eval()
    logger.info(model)

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.train_data = train_data
    trainer.eval_collector.data_collect(train_data)
    gc.collect()
    empty_cache()

    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=config.get("show_progress", True)
    )

    logger.info(set_color("Inference Test Result", "yellow") + f": {test_result}")

    return test_result


def _setup_model(config, train_data, dataset_obj, logger):
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    empty_cache()
    return model


def _run_evaluation(trainer, test_data, saved, config):
    gc.collect()
    empty_cache()

    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )
    return test_result


def _log_final_results(logger, best_valid, test_result, config):
    env_tb = get_environment(config)
    logger.info("Environment:\n" + env_tb.draw())
    logger.info(set_color("Best Valid", "yellow") + f": {best_valid}")
    logger.info(set_color("Test Result", "yellow") + f": {test_result}")


def _cleanup(config, queue, result):
    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)
