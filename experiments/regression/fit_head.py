import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../../"))
from MMSA.__main__ import parse_args_custom 
from MMSA.run import MMSA_run
from MMSA.config import *


if __name__ == "__main__":
    # path_to_config = "MMSA/src/MMSA/config/regression"
    # mdl_cfg = "mult_base.json"
    cmd_args = parse_args_custom()
    # cmd_args.config = os.path.join(os.getcwd(), cmd_args.config)
    # get the custom config file
    cfg = get_config_regression(
        model_name=cmd_args.model,
        dataset_name=cmd_args.dataset,
        config_file=cmd_args.config,
    )
    # Run experiment
    MMSA_run(
        model_name=cmd_args.model,
        dataset_name=cmd_args.dataset,
        config_file=cmd_args.config,
        seeds=cmd_args.seeds,
        is_tune=cmd_args.tune,
        tune_times=cmd_args.tune_times,
        feature_T=cmd_args.feature_T,
        feature_A=cmd_args.feature_A,
        feature_V=cmd_args.feature_V,
        model_save_dir=cmd_args.model_save_dir,
        res_save_dir=cmd_args.res_save_dir,
        log_dir=cmd_args.log_dir,
        gpu_ids=cmd_args.gpu_ids,
        num_workers=cmd_args.num_workers,
        verbose_level=cmd_args.verbose,
        exp_name=cmd_args.exp_name,
    )