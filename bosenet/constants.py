
"""Constants."""

import functools
import jax

from kfac_ferminet_alpha import utils as kfac_utils

# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)

# Shortcut for kfac utils
psum = functools.partial(kfac_utils.psum_if_pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(kfac_utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)

