import os
from pathlib import Path

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import to_absolute_path
from hydra import initialize_config_dir
from omegaconf import OmegaConf

""" BEGIN HYDRA HELPERS """
def hydra_compose(
    config_path: str = "configs",      # directory containing your YAML tree
    config_name: str = "eval",         # root config file, e.g. configs/eval.yaml
    overrides: list[str] | None = None # hydra-style CLI overrides
):
    """Compose a Hydra config in notebooks (no decorator)."""
    overrides = overrides or []
    # Allow re-compose in notebooks
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    # Initialize relative to the cwd
    config_dir = to_absolute_path(config_path)
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg

def resolve(cfg):
    """Return a fully resolved (interpolations expanded) copy."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

def ensure_outdir_and_optionally_chdir(cfg, chdir: bool = False):
    """Mimic hydra.run.dir behavior manually in notebooks."""
    out_dir = cfg.eval.logging.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if chdir:
        os.chdir(out_dir)
    return out_dir
""" END HYDRA HELPERS """