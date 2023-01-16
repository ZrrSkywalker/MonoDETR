import os
from typing import Optional
import logging
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None, best_result=None, best_epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, DDP):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state, 'best_result': best_result, 'best_epoch': best_epoch}


def save_checkpoint(state, filename):
    filename = f'{filename}.pth'
    logging.getLogger().info(f'Saving the checkpoint to {filename}...')
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, map_location, logger: Optional[logging.Logger] = None):
    if os.path.isfile(filename):
        if logger:
            logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)
        best_result = checkpoint.get('best_result', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            if isinstance(model, DDP):
                model.module.load_state_dict(checkpoint['model_state'], strict=False)
            else:
                model.load_state_dict(checkpoint['model_state'], strict=False)
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if logger:
            logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch, best_result, best_epoch
