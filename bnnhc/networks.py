
"""Implementation of Bosonet Helium Cluster Neural Network in JAX."""

from typing import Tuple

from bnnhc import curvature_tags_and_blocks

import jax
import jax.numpy as jnp

BoseLayers = Tuple[int,...]

def init_bosenet_params(
    key: jnp.ndarray, 
    hidden_dims: BoseLayers = ((64,16),(64,16)),
    np: int = 3,
    dim: int = 3,
    num_orbitals: int = 8
):
  """ Creates the initial set of variational parameters

  Args:
    key: RNG key
    hidden_dims: neural networks architecture 
         (sizes of each stream for each layer)
    np: number of particles
    dim: number of spatial dimensions
    num_orbitals: number of symmetric functions 
  """
  
  in_dim   = 2*(dim+1)
  dims_in  = ( [in_dim]+[ hdim[0]+hdim[1] for hdim in hidden_dims ] )
  dims_out = [ hdim[0] for hdim in hidden_dims ] 
  dims_two = [dim+1] + [hdim[1] for hdim in hidden_dims]
  
  params = {
      'single': [ {} for _ in range(len(hidden_dims)) ],
      'double': [ {} for _ in range(len(hidden_dims)) ],
      'orbital': {},
      'envelope': {},
      'omega': [],
  }

  # Exponential decay factors
  # params['envelope']['a'] = 0.10*jnp.ones(shape=(np*num_orbitals))
  params['envelope']['a'] = 0.10*jnp.ones(shape=(num_orbitals))

  # Neural networks weights and biases
  for i in range(len(hidden_dims)):

    key, subkey = jax.random.split(key)
    shape = ( dims_in[i], dims_out[i] )
    scale = jnp.sqrt( float(dims_in[i]) )
    params['single'][i]['w'] = ( jax.random.normal(subkey, shape=shape) / scale )

    key, subkey = jax.random.split(key)
    shape = ( dims_out[i], )
    params['single'][i]['b'] = jax.random.normal(subkey, shape=shape)
    
    key, subkey = jax.random.split( key )
    shape = (dims_two[i], dims_two[i+1])
    scale = jnp.sqrt(float(dims_two[i]))
    params['double'][i]['w'] = ( jax.random.normal(subkey, shape=shape) / scale )

    key, subkey = jax.random.split(key)
    shape = (dims_two[i+1],)
    params['double'][i]['b'] = jax.random.normal(subkey, shape=shape)


  # Output layer weights and biases
  key, subkey = jax.random.split(key)
  # shape = ( dims_in[-1], np*num_orbitals )
  shape = ( dims_in[-1], num_orbitals )
  scale = jnp.sqrt(float(dims_in[-1]))
  params['orbital']['w'] = jax.random.normal(subkey, shape=shape) / scale

  key, subkey = jax.random.split(key)
  # shape = ( np*num_orbitals, )
  shape = ( num_orbitals, )
  params['orbital']['b'] = jax.random.normal(subkey, shape=shape)

  # Linear combinations coefficients
  key, subkey = jax.random.split(key)
  params['omega'] = jax.random.normal( subkey, shape=(num_orbitals,) )
  
  return params


def construct_input_features(
    x: jnp.ndarray,
    dim: int = 3 
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """ Construct the input features f^0_i of the network for each 
      particle i

  Args:
    x: array with a set of particles positions [shape->(np*dim)]
      
  Returns:
    p: array of coordinates relative to the center of mass [shape->(np,dim)]
    r: array of distance to the center of mass [shape->(np,1)]
    pp: matrix of relative coordinates for each dimension [shape->(np,np,dim)]
    dr: matrix of relative distances [shape->(np,np,1)]
  """
  
  p = jnp.reshape(x, [-1, dim])
  p = p - jnp.mean(p, axis=0, keepdims=True)
  r = jnp.linalg.norm(p, axis=-1, keepdims=True)

  pp = (
    jnp.reshape(x, [1, -1, dim]) - jnp.reshape(x, [-1, 1, dim]) )

  n = pp.shape[0]

  dr = (
    jnp.linalg.norm(pp + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))

  return p, r, pp, dr


def construct_symmetric_features(h_one, h_two):
  """ Construct the symmetric features using a crossover
      information from the single-atom and two-atom streams.

  Args: 
    h_one: single-atom stream
    h_two: two-atom stream 

  Returns: symmetric features for each particle  
  """

#  g_one = jnp.mean( h_one, axis=0, keepdims=True )
  g_two = [ jnp.mean( h_two, axis=0 ) ]

#  return jnp.concatenate([h_one-g_one]+g_two, axis=1)
  return jnp.concatenate([h_one]+g_two, axis=1)


def linear_layer(x, w, b=None):
  """Evaluates a linear layer, x w + b.

  Args:
    x: inputs.
    w: weights.
    b: optional bias. Only x w is computed if b is None.

  Returns:
    x w + b if b is given, x w otherwise.
  """
  y = jnp.dot(x, w)
  y = y + b if b is not None else y
  return curvature_tags_and_blocks.register_repeated_dense(y, x, w, b)


vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)


def neural_network(h_one, h_two, params):
  """ Computes the feedforward neural network outputs.

  Args:
    h_one: single-atom input stream
    h_two: two-atom input stream
    params: variational parameters

  Returns: neural networks outputs
  """

  residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
  for i in range(len(params['single'])):
    h_one_in = construct_symmetric_features(h_one, h_two) 

    h_one_next = jnp.tanh(linear_layer(h_one_in, **params['single'][i]))
    h_two_next = jnp.tanh(vmap_linear_layer(h_two, params['double'][i]['w'], 
                                                   params['double'][i]['b']))

    h_one = residual(h_one, h_one_next)
    h_two = residual(h_two, h_two_next)

  h = construct_symmetric_features(h_one, h_two)
  h_next = linear_layer(h, **params['orbital'])
  h = residual(h, h_next)

  h = jax.nn.sigmoid(h)

  return h


def gaussian_envelope(r, params):
  """ Computes the exponential decay as the 
      asymptotic behaviour for r going to infinity

  Args: 
    r: distances to the center of mass of each particle
    params: variational decay parameters

  Returns: log of the gaussian envelope for each combination of
           a particle-parameter
  """
  env = -jnp.abs(r**2*params)
  return env


def mcmillan_envelope(dr, params):
  """ Computes the McMillan envelope of the Ansatz

  Args:
    r: relative distances for each pair of particles
    params: McMillan parameter (fixed)

  Returns: log of the McMillan envelope
  """
  odr = (1.0 - jnp.eye(dr.shape[0])) / (dr + jnp.eye(dr.shape[0]))
  odr = jnp.triu( odr, k=1 ) 
  env = -jnp.abs(jnp.sum( odr**5 )*params)
  return env


def hardsphere_envelope(dr):
  """ Computes a very low log probability (virtually zero) if at least one
      pair of particles are closer than a fixed treshold.
  
  Args:
    r: relative distances for each pair of particles
  """
 # cutoff = 0.6 - dr - jnp.eye(dr.shape[0])
  cutoff = 0.64 - dr - jnp.eye(dr.shape[0])
  return -1e20*jnp.sum(jnp.heaviside(cutoff, 0.5))


def bosenet_orbital(params, x):
  """ Computes the network outputs and envelopes to
      construct the symmetric functions

  Args: 
    params: variational parameters
    x: set of particle positions

  Returns: 
    orbital: NN outputs
    envelopes: log gaussian decays
  """
  v, r, pp, dr = x 

  h_one = jnp.concatenate((r,v), axis=-1)
  h_two = jnp.concatenate((dr[...,None], pp), axis=-1)

  envelope = gaussian_envelope(r, params['envelope']['a'])

  orbitals = neural_network(h_one, h_two, params)

  return orbitals, envelope


def bosenet(params, x):
  """ Computes the log of the trial wave function.

  Args: 
    params: variational parameters
    x: set of particle positions
  """
  
  f = construct_input_features(x)

  mcm = mcmillan_envelope(f[-1], jnp.ones(1)*0.50) + hardsphere_envelope(f[-1])

  orb, env = bosenet_orbital(params, f)

  n = orb.shape[0]

  #Suposses that orb>0, since orb=sigmoid(h)
  # logphi = jnp.reshape(jnp.log(orb)+env, [n,n,-1])
  # maxlogphi = jnp.max(logphi, axis=1, keepdims=True)
  logphi = jnp.log(orb)+env

  # phibar = jnp.exp(logphi-maxlogphi)
  # chibar = jnp.prod(jnp.sum(phibar, axis=1), axis=0)

  # logchi = jnp.sum(maxlogphi, axis=(0,1))+jnp.log(chibar)
  logchi = jnp.sum(logphi, axis=0)
  maxlogchi = jnp.max(logchi)
  
  zeta = params['omega']*jnp.exp(logchi-maxlogchi)

  logprob = mcm[0] + maxlogchi + jnp.log(jnp.abs(jnp.sum(zeta)))

  return logprob


def bosenet_vmc(params, x):
  """ Computes the log of the trial wave function for VMC simulations.

  Args: 
    params: variational parameters
    x: set of particle positions
  """

  f = construct_input_features(x)

  mcm = mcmillan_envelope(f[-1], jnp.ones(1)*0.50)
  
  orb, env = bosenet_orbital(params, f)

  n = orb.shape[0]

  #Suposses that orb>0, since orb=sigmoid(h)
  # logphi = jnp.reshape(jnp.log(orb)+env, [n,n,-1])
  # maxlogphi = jnp.max(logphi, axis=1, keepdims=True)
  logphi = jnp.log(orb)+env #

  # phibar = jnp.exp(logphi-maxlogphi)
  # chibar = jnp.prod(jnp.sum(phibar, axis=1), axis=0)

  # logchi = jnp.sum(maxlogphi, axis=(0,1))+jnp.log(chibar)
  logchi = jnp.sum(logphi, axis=0) #
  maxlogchi = jnp.max(logchi)
  
  zeta = params['omega']*jnp.exp(logchi-maxlogchi)

  logprob = mcm[0] + maxlogchi + jnp.log(jnp.abs(jnp.sum(zeta)))

  return logprob

