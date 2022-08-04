
"""Implementation of Bosonic Neural Network in JAX."""

from typing import Tuple

from bosenet import curvature_tags_and_blocks

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
  """
  """
  
  in_dim   = 3*(dim+1)
  dims_in  = ( [in_dim]+[ 2*hdim[0]+hdim[1] for hdim in hidden_dims ] )
  dims_out = [ hdim[0] for hdim in hidden_dims ] 
  dims_two = [dim+1] + [hdim[1] for hdim in hidden_dims]
  
  params = {
      'single': [ {} for _ in range(len(hidden_dims)) ],
      'double': [ {} for _ in range(len(hidden_dims)) ],
      'orbital': {},
      'envelope': {},
      'omega': [],
  }

  params['envelope']['a'] = 0.10*jnp.ones(shape=(np*num_orbitals))

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


  key, subkey = jax.random.split(key)
  shape = ( dims_in[-1], np*num_orbitals )
  scale = jnp.sqrt(float(dims_in[-1]))
  params['orbital']['w'] = jax.random.normal(subkey, shape=shape) / scale

  key, subkey = jax.random.split(key)
  shape = ( np*num_orbitals, )
  params['orbital']['b'] = jax.random.normal(subkey, shape=shape)

  key, subkey = jax.random.split(key)
  params['omega'] = jax.random.normal( subkey, shape=(num_orbitals,) )
  
  return params


def construct_input_features(
    x: jnp.ndarray,
    dim: int = 3 
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """
  Args:
    x (np*dim)
      
  Returns:
    p  (np,dim)
    r  (np,1)
    pp (np,np,dim)
    dr (np,np,1)
  """

  p = jnp.reshape(x, [-1, 1, dim])
  r = jnp.linalg.norm(p, axis=2, keepdims=True)

  pp = (
    jnp.reshape(x, [1, -1, dim]) - jnp.reshape(x, [-1, 1, dim]) )

  n = pp.shape[0]

  dr = (
    jnp.linalg.norm(pp + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))

  return p, r, pp, dr


def construct_symmetric_features(h_one, h_two):
  """
  """

  g_one = [ jnp.mean( h_one, axis=0, keepdims=True ) ]
  g_one = [ jnp.tile( *g_one, [h_one.shape[0],1] ) ]

  g_two = [ jnp.mean( h_two, axis=0 ) ]

  return jnp.concatenate([h_one]+g_one+g_two, axis=1)


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
  """
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
  """
  """
  env = -jnp.abs(r**2*params)
  return env.squeeze(1) 


def mcmillan_envelope( dr, params ):
  """
  """
  odr = (1.0 - jnp.eye(dr.shape[0])) / (dr + jnp.eye(dr.shape[0]))
  odr = jnp.triu( odr, k=1 ) 
  env = -jnp.abs(jnp.sum( odr**5 )*params)
  return env


def bosenet_orbital(params, x):
  """
  """
  v, r, pp, dr = construct_input_features(x)

  h_one = jnp.concatenate((r,v), axis=2)               # Shape (np,1,dim+1)
  h_one = jnp.reshape(h_one, [jnp.shape(r)[0], -1])    # Shape (np,dim+1)
  h_two = jnp.concatenate((dr[...,None], pp), axis=2)  # Shape (np,np,dim+2)

  envelope = gaussian_envelope(r, params['envelope']['a'])

  orbitals = neural_network(h_one, h_two, params)

  mcmillan = mcmillan_envelope(dr, jnp.ones(1)*0.50)

  return orbitals, envelope, mcmillan


def bosenet(params, x):
  """
  Returns ln psi
  """
  orb, env, mcm = bosenet_orbital(params, x)

  n = orb.shape[0]

  #Suposses that orb>0, since orb=sigmoid(h)
  logphi = jnp.reshape(jnp.log(orb)+env, [n,n,-1])

  maxlogphi = jnp.max(logphi, axis=1, keepdims=True)

  phibar = jnp.exp(logphi-maxlogphi)
  chibar = jnp.prod(jnp.sum(phibar, axis=1), axis=0)

  logchi = jnp.sum(maxlogphi, axis=(0,1))+jnp.log(chibar)
  maxlogchi = jnp.max(logchi)

  zeta = params['omega']*jnp.exp(logchi-maxlogchi)

  logprob = mcm[0] + maxlogchi + jnp.log(jnp.abs(jnp.sum(zeta)))

  return logprob

