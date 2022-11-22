from lib.models.monodetr import build_monodetr


def build_model(model_cfg, loss_cfg):
    return build_monodetr(model_cfg, loss_cfg)
