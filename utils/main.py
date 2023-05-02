# standard libraries
import argparse
import os
import sys

# external libraries
import numpy as np
import torch
from loguru import logger

# sys.path.append("./anomaly_detector")
import pipeline
import trainer
# internal libraries
from utils import cmd_args_utils


@logger.catch
def main(parser):
    torch.multiprocessing.set_start_method('spawn')  # good solution !!!!
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.exp_folder):
        os.system(f"mkdir -p {args.exp_folder}")

    pl = pipeline.SourceExtractionPL(parser)
  
    trainer.train(pl.model, pl.train, pl.optimizer, pl.config, pl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cmd_args_utils.add_common_flags(parser)
    logger.info(f"python {' '.join(sys.argv)}")
    
    main(parser)
