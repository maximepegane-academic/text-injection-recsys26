import argparse
import yaml

from pathlib import Path
from google.cloud import storage
from time import perf_counter
from utils import add_args_to_parser, setup_run_environment
from quick_experiment import run_recbole, run_inference
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    start = perf_counter()
    if not Path("config.yaml").is_file():
        try:
            storage.Client().get_bucket(BOOTSTRAP_BUCKET).blob(
                BOOTSTRAP_CONFIG_PATH
            ).download_to_filename("config.yaml")
        except Exception as e:
            print(f"Failed to download config: {e}")

    with open("config.yaml") as stream:
        base_config = yaml.safe_load(stream) or {}

    parser = argparse.ArgumentParser()
    parser = add_args_to_parser(parser, base_config)
    args, _ = parser.parse_known_args()

    base_config.update(vars(args))

    config = setup_run_environment(base_config)

    if config.get("inference_only", False):
        print(f"Running inference only using model: {config['model_path']}")
        res = run_inference(
            model_path=config["model_path"],
            config_dict=config  # Passes your args overrides (like device="cuda")
        )
    else:
        res = run_recbole(
            config["model"],
            config["dataset"],
            config_dict=config,
        )

    end = perf_counter()
    elapsed = end - start
    print(f"Time taken: {elapsed:.6f} seconds")

