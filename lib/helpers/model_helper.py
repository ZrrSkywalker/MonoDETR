from lib.models.monodetr import build_monodetr


def build_model(cfg):
    return build_monodetr(cfg)
