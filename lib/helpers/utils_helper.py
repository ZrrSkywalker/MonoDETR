import torch
import numpy as np
import logging
import random


def create_logger(log_file, rank=0):
    log_format = '%(asctime)s %(levelname)5s' + f' rank={rank} ' + '%(filename)-20s line=%(lineno)3s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger(__name__)
    logger.addHandler(console)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
