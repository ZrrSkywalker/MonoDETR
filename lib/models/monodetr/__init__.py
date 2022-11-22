from .monodetr import build


def build_monodetr(model_cfg, loss_cfg):
    return build(model_cfg, loss_cfg)
