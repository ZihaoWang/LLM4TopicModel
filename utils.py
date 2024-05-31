import os
import logging
import time
import pickle
from typing import Callable, TypeVar, Any, Union
from argparse import Namespace
from collections.abc import Iterable

from args import get_args

def init_env_args_logging() -> Namespace:
    '''
    Setting up environments and the log, parsing arguments, 

            Return:
                    args: all parsed arguments.
    '''

    os.environ['OPENBLAS_NUM_THREADS'] = '56'

    '''
    secrets = {}
    with open("secrets.txt", "r") as f_src:
        for line in f_src:
            name, secret = line.strip().split("=")
            secrets[name] = secret

    from huggingface_hub import login
    login(token=secrets["huggingface_token"])
    '''

    args = get_args()
    
    log_path = os.path.join(args.log_root, "topic_llama3.log")
    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_path,
        filemode="w"
    )
    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info(f"Saving the log at: {log_path}\nParameters:\n--------------------------------\n")

    summary_dict = vars(args)
    for k, v in summary_dict.items():
        logging.info(f"{k} = {v}\n")

    logging.info("End of parameters.\n------------------------------------\n")

    for root in ["tmp_root", "log_root", "result_root"]:
        os.makedirs(summary_dict[root], exist_ok = True)

    return args

