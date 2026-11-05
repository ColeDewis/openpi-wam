# !/usr/bin/env python3

import jax
from jax import nn
import jax.numpy as jnp

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.common.encoding import StateEncoder
from serl_launcher.vision.data_augmentations import batched_random_crop

##############################################################################
# Upstream hil-serl agent factories. The image-based paths are unchanged; the
# low-dim factories below are the additions needed for the `rltoken` and
# `timm` encoder options, where a frozen feature extractor has already turned
# the observation into a vector stored under the `state` key.
##############################################################################


def make_bc_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
):
    return BCAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )


def make_sac_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    soft_target_update_rate=0.005,
):
    agent = SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        soft_target_update_rate=soft_target_update_rate,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_bc_lowdim_agent(
    seed,
    sample_obs,
    sample_action,
):
    """HG-Dagger agent for a pre-encoded observation vector (``{"state": ...}``)."""
    return BCAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        encoder_def=StateEncoder(),
        encoder_type="state",  # not resnet-*: skip the ImageNet weight load
        image_keys=("state",),
        augmentation_function=None,
    )


def make_sac_lowdim_agent(
    seed,
    sample_obs,
    sample_action,
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    soft_target_update_rate=0.005,
):
    """RLPD agent for a pre-encoded observation vector (``{"state": ...}``)."""
    return SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_def=StateEncoder(),
        encoder_type="state",  # not resnet-*: skip the ImageNet weight load
        image_keys=("state",),
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        soft_target_update_rate=soft_target_update_rate,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=None,
    )


def make_sac_hybrid_agent(
    seed,
    sample_obs,
    sample_action,
    *,
    encoder_kind="hilserl",
    image_keys=("front_image", "wrist_image"),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    soft_target_update_rate=0.005,
    target_intervention_rate=0.05,
):
    """hil-serl's ``SACAgentHybridSingleArm`` (RLPD with a DQN gripper).

    The first ``action_dim - 1`` dims are continuous (SAC); the last dim is a
    discrete gripper in {-1, 0, +1} learned by a 3-way DQN "grasp critic".
    """
    mlp = lambda: {"activations": nn.tanh, "use_layer_norm": True, "hidden_dims": [256, 256]}  # noqa: E731
    kwargs = dict(
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs=mlp(),
        cost_critic_network_kwargs=mlp(),
        grasp_critic_network_kwargs=mlp(),
        policy_network_kwargs=mlp(),
        temperature_init=1e-2,
        target_intervention_rate=target_intervention_rate,
        discount=discount,
        soft_target_update_rate=soft_target_update_rate,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
    )
    if encoder_kind == "hilserl":
        return SACAgentHybridSingleArm.create_pixels(
            jax.random.PRNGKey(seed),
            sample_obs,
            sample_action,
            encoder_type=encoder_type,
            use_proprio=True,
            image_keys=image_keys,
            augmentation_function=make_batch_augmentation_func(image_keys),
            **kwargs,
        )
    return SACAgentHybridSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_def=StateEncoder(),
        encoder_type="state",  # not resnet-*: skip the ImageNet weight load
        image_keys=("state",),
        augmentation_function=None,
        **kwargs,
    )


def make_agent(
    algorithm,
    seed,
    sample_obs,
    sample_action,
    *,
    encoder_kind="hilserl",
    image_keys=("front_image", "wrist_image"),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    soft_target_update_rate=0.005,
    target_intervention_rate=0.05,
):
    """Build the RLPD (``sac``) or HG-Dagger (``bc``) agent for an encoder kind.

    encoder_kind:
        hilserl  -- images are encoded inside the agent by a ResNet-10, as upstream.
        rltoken  -- observations are frozen VLA RL-token features (``{"state"}``).
        timm     -- observations are frozen timm features (``{"state"}``).
    """
    if algorithm not in ("sac", "sac_hybrid", "bc"):
        raise ValueError(f"unknown algorithm {algorithm!r} (expected 'sac', 'sac_hybrid' or 'bc')")
    if algorithm == "sac_hybrid":
        return make_sac_hybrid_agent(
            seed,
            sample_obs,
            sample_action,
            encoder_kind=encoder_kind,
            image_keys=image_keys,
            encoder_type=encoder_type,
            reward_bias=reward_bias,
            target_entropy=target_entropy,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            target_intervention_rate=target_intervention_rate,
        )
    if encoder_kind == "hilserl":
        if algorithm == "bc":
            return make_bc_agent(seed, sample_obs, sample_action, image_keys, encoder_type)
        return make_sac_pixel_agent(
            seed,
            sample_obs,
            sample_action,
            image_keys,
            encoder_type,
            reward_bias,
            target_entropy,
            discount,
            soft_target_update_rate,
        )
    if encoder_kind in ("rltoken", "timm"):
        if algorithm == "bc":
            return make_bc_lowdim_agent(seed, sample_obs, sample_action)
        return make_sac_lowdim_agent(
            seed, sample_obs, sample_action, reward_bias, target_entropy, discount, soft_target_update_rate
        )
    raise ValueError(f"unknown encoder_kind {encoder_kind!r}")


def linear_schedule(step):
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000

    linear_step = jnp.minimum(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value


def make_batch_augmentation_func(image_keys) -> callable:
    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    def augment_batch(batch, rng):
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        return batch

    return augment_batch
