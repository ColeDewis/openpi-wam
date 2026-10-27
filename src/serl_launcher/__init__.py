"""Vendored subset of rail-berkeley/hil-serl's ``serl_launcher``.

The files here are copied from hil-serl with only the changes needed to drop
runtime dependencies that this repo does not carry (agentlace, wandb,
tensorflow, distrax). Keeping the package name and the internal import paths
identical to upstream means the two trees can be read side by side:

    https://github.com/rail-berkeley/hil-serl/tree/main/serl_launcher/serl_launcher

RLPD (``agents/continuous/sac.py``) and HG-Dagger (``agents/continuous/bc.py``)
are the two algorithms this package exists to provide.
"""
