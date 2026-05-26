import os
import sys
import yaml
import warnings
import datetime
import logging
import torch
import polars as pl
from pathlib import Path
from google.cloud import storage
from flatdict import FlatDict

DEFAULT_BUCKET = "dataproc-research-sandbox"
DEFAULT_PROJECT = "research-254009"
DEFAULT_ROOT_PREFIX = "lucky-6203-consumer-prediction-under-promotion"

try:
    from names_generator import generate_name
except ImportError:

    def generate_name(style="hyphen"):
        return "experiment"


def human_format(num, round_to=0):
    if num is None or num < 1:
        return "None"
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num = num / 1000.0
    suffix = ["", "K", "M", "G", "T"][magnitude]
    return "{:.{}f}{}".format(round(num, round_to), round_to, suffix)


def flatten_stringify_config(config):
    config = FlatDict(config)
    for k, v in config.items():
        if not isinstance(v, (float, int, str)):
            config[k] = str(v)
    return config


def _parse_cli_args():
    args = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.strip("-").split("=", 1)
            args[key] = value
    return args


def get_default_args(model_name):
    try:
        current_path = os.path.dirname(os.path.realpath(__file__))
        model_init_file = os.path.join(
            current_path, f"../properties/model/{model_name}.yaml"
        )

        if os.path.exists(model_init_file):
            with open(model_init_file, "r") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load default args for {model_name}: {e}")
    return {}


def get_info_from_metric_name(metric_name):
    m_type = "contribution" if "contribution" in metric_name else "True"

    if "explore" in metric_name:
        subset = "explore"
    elif "repeat" in metric_name:
        subset = "repeat"
    else:
        subset = "all"

    parts = metric_name.split("@")
    k = parts[1] if len(parts) > 1 else 0
    clean_name = parts[0].split("_")[-1]

    return clean_name, m_type, subset, k


def setup_run_environment(config):
    job_id = os.environ.get("CLOUD_ML_JOB_ID")
    trial_id = os.environ.get("CLOUD_ML_TRIAL_ID", "default")
    is_cloud = bool(job_id)
    is_hyperparam_search = bool(trial_id)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    hr_sample = human_format(config.get("number_random_uid_sample", 0))
    bucket_name = config.get("GCS_BUCKET_NAME", DEFAULT_BUCKET)
    root_prefix = config.get("GCS_ROOT_PREFIX", DEFAULT_ROOT_PREFIX)

    if is_cloud:
        run_name = f"{config['model']}_{config['dataset']}_{hr_sample}_job_{job_id}"
        if is_hyperparam_search:
            run_name += f"_{trial_id}"
    else:
        readable_name = generate_name(style="hyphen")
        run_name = f"{config['model']}_{config['dataset']}_{hr_sample}_{readable_name}_{timestamp}"

    local_root = config.get("data_path", "experiments")
    run_dir = os.path.join(local_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    remote_prefix = (
        f"{root_prefix}/Run_Metrics/"
        f"{config['model']}/{config['dataset']}/{hr_sample}/"
    )
    if is_cloud:
        remote_prefix += f"{job_id}/"
        if is_hyperparam_search:
            remote_prefix += f"{trial_id}/"
    remote_prefix += run_name
    config.update(
        {
            "run_name": run_name,
            "run_dir": run_dir,
            "checkpoint_dir": run_dir,
            "log_bucket": config.get(
                "log_bucket", not bool(os.environ.get("IS_LOCAL_RUN"))
            ),
            "gcs_bucket": bucket_name,
            "gcs_prefix": remote_prefix,
            "gcs_root_prefix": root_prefix,
            "is_hyperparam_search": is_hyperparam_search,
            "wandb_project": config.get("wandb_project", "RecBole-Experiments"),
            "wandb_run_name": run_name,
            "log_wandb": config.get("log_wandb", False),
        }
    )

    return config


def get_run_summary_dfs(model, result, config):
    test_dict = result.get("test_result", {})
    valid_dict = result.get("best_valid_result", {})

    meta = {
        "job_id": os.environ.get("CLOUD_ML_JOB_ID", "local"),
        "trial_id": os.environ.get("CLOUD_ML_TRIAL_ID", "local"),
        "date": datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": config["model"],
        "dataset": config["dataset"],
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_users": config["number_random_uid_sample"],
    }

    metric_rows = []

    def add_metrics(metrics_dict, phase):
        for metric_str, val in metrics_dict.items():
            name, type_, subset, k = get_info_from_metric_name(metric_str)
            row = meta.copy()
            row.update(
                {
                    "phase": phase,
                    "metric_type": type_,
                    "metric_name": name,
                    "k": int(k),
                    "subset": subset,
                    "metric_value": float(val),
                }
            )
            metric_rows.append(row)

    add_metrics(test_dict, "test")
    add_metrics(valid_dict, "valid")

    metric_df = pl.DataFrame(metric_rows)

    input_args = _parse_cli_args()

    base_hparams = get_default_args(config["model"]).keys()
    for param in base_hparams:
        if param in config:
            input_args[param] = config[param]

    if hasattr(model, "learning_rate"):
        input_args["learning_rate"] = config["learning_rate"]

    ignore_keys = {
        "model",
        "dataset",
        "number_random_uid_sample",
        "epochs",
        "eval_batch_size",
        "train_batch_size",
        "track_experiment",
        "experiment_name",
    }

    hparam_rows = []
    for k, v in input_args.items():
        if k not in ignore_keys:
            row = meta.copy()
            row.update({"hparam": k, "hparam_value": str(v)})
            hparam_rows.append(row)

    hparam_df = pl.DataFrame(hparam_rows)

    return metric_df, hparam_df


def save_experiment_results(model, result, config, trainer):
    if config["local_rank"] != 0:
        return

    metric_df, hparam_df = get_run_summary_dfs(model, result, config)

    print("\nMetrics summary:")
    print(metric_df)
    print("\nHparams summary:")
    print(hparam_df)

    run_dir = config.get("run_dir", config.get("data_path", "."))
    base_name = config.get("run_name", "experiment")

    metric_csv_path = os.path.join(run_dir, f"{base_name}_metrics.csv")
    param_csv_path = os.path.join(run_dir, f"{base_name}_params.csv")

    metric_df.write_csv(metric_csv_path)
    hparam_df.write_csv(param_csv_path)

    result["metric_csv_path"] = metric_csv_path
    result["param_csv_path"] = param_csv_path
    result["model_path"] = trainer.saved_model_file
    result["predictions_path"] = trainer.saved_prediction_path


def add_args_to_parser(parser, config):
    parser.add_argument("--model", "-m", type=str, default="BPR")
    parser.add_argument("--dataset", "-d", type=str, default="dummy")
    parser.add_argument("--config_files", type=str, default=None)

    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="5678")
    parser.add_argument("--world_size", type=int, default=-1)
    parser.add_argument("--group_offset", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=config.get("epochs", 40))
    parser.add_argument("--eval_mode", type=float, default=10)
    parser.add_argument(
        "--train_batch_size", type=int, default=config.get("train_batch_size", 4096)
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=config.get("eval_batch_size", 4096)
    )
    parser.add_argument(
        "--learning_rate", type=float, default=config.get("learning_rate", 0.01)
    )
    parser.add_argument(
        "--embedding_size", type=int, default=config.get("embedding_size", 64)
    )
    parser.add_argument(
        "--hidden_size", type=int, default=config.get("hidden_size", 64)
    )
    parser.add_argument("--inner_size", type=int, default=config.get("inner_size", 256))
    parser.add_argument("--n_layers", type=int, default=config.get("n_layers", 2))
    parser.add_argument("--n_heads", type=int, default=config.get("n_heads", 2))

    parser.add_argument("--n_v", type=int, default=config.get("n_v", 4))
    parser.add_argument("--n_h", type=int, default=config.get("n_h", 4))

    parser.add_argument("--k_interests", type=int, default=config.get("k_interests"))
    parser.add_argument(
        "--knn_method", type=str, default=config.get("knn_method", "full")
    )
    parser.add_argument(
        "--num_neighbors", type=int, default=config.get("num_neighbors", 200)
    )
    parser.add_argument(
        "--within_group_decay",
        type=float,
        default=config.get("within_group_decay", 0.9),
    )
    parser.add_argument(
        "--global_decay", type=float, default=config.get("global_decay", 0.9)
    )
    parser.add_argument("--alpha", type=float, default=config.get("alpha", 0.5))

    parser.add_argument(
        "--num_negative_sample", default=config.get("train_neg_sample_args", 1)
    )
    parser.add_argument(
        "--number_random_uid_sample",
        type=int,
        default=config.get("number_random_uid_sample", -1),
    )

    parser.add_argument(
        "--inference_only",
        type=bool,
        default=config.get("inference_only", False),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=config.get("model_path"),
    )

    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--location", type=str, default="europe-west2")
    parser.add_argument("--project", type=str, default="research-254009")
    parser.add_argument("--track_experiment", type=bool, default=False)
    parser.add_argument("--log_wandb", type=bool, default=False)

    return parser
