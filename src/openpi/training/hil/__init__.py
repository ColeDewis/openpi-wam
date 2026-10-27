"""RLPD / HG-Dagger for this repo.

RLPD and HG-Dagger themselves are vendored from hil-serl under
``src/serl_launcher``. This package supplies the WAM-specific glue:

    encoders.py    -- choose hilserl / rltoken / timm observations
    transitions.py -- the actor -> learner pickle handoff
    learner.py     -- the shared training loop used by both scripts
"""
