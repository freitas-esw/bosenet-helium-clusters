
"""Constants."""

import functools
import jax

import kfac_jax
#from kfac_ferminet_alpha import utils as kfac_utils

# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)

# Shortcut for kfac utils
#pmean = functools.partial(kfac_utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(kfac_jax.utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)

p_split = kfac_jax.utils.p_split

replicate = kfac_jax.utils.replicate_all_local_devices

broadcast = kfac_jax.utils.broadcast_all_local_devices

shard_key = kfac_jax.utils.make_different_rng_key_on_all_devices
