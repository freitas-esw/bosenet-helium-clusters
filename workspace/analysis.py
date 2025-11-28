import os
import jax

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import math as mt
import jax.numpy as jnp

########################################################
########################################################

def read_energy_data(label):
  fns = [fn for fn in os.listdir() if fn.startswith(label)]
  data = pd.DataFrame()
  for fn in fns:
    data = pd.concat([data, pd.read_csv(fn)])
  data = data.sort_values(by='step', ascending=True)
  return data

########################################################
########################################################

def factorize(num):
  """
  Factorize an integer number.
  """ 
  return [n for n in range(1, num + 1) if num % n == 0]

def estimation(ene):
  ave = jnp.mean(ene)
  if ene.size > 1:
    std = jnp.sqrt(jnp.sum((ene-jnp.mean(ene))**2)/(ene.size*(ene.size-1)))
  else:
    std = jnp.nan
  return ave, std

def blocking(data, block_sizes):
  vsd = jnp.zeros(0)
  nsteps = jnp.size(data)
  for j in block_sizes:
    nblocks = nsteps//j
    data_block = jnp.mean(data.reshape([nblocks, j]), axis=1)
    ave_dt, std_dt = estimation(data_block)
    vsd = jnp.append(vsd, std_dt)
  return ave_dt, vsd
  
def energy_estimation(data):
  bs = factorize(data['step'].values[-1] + 1)[2:-1]
  ene = data['energy'].values
  ene, std = blocking(ene, bs)
  return bs, ene, std

def weighted_averages(data, alpha):
  Wdat = np.zeros_like(data.values)
  Wvar = np.zeros_like(data.values)
  Wdat[0] = data.values[0] 
  
  for i in range(1, len(data.values[1:]) + 1):
    incr = alpha * (data.values[i] - Wdat[i-1])
    Wdat[i] = Wdat[i-1] + incr
    Wvar[i] = (1 - alpha) * (Wvar[i-1] + incr**2 / alpha)

  return Wdat, Wvar

########################################################
########################################################

def get_samples(path = '.', ndim = 3):
  f = open('samples.npy', 'rb')
  pos = np.load(f)
  while True:
#  for _ in range(4):
    try:
      pos = np.concatenate([pos, np.load(f)], axis=0)
    except:
      break
  f.close()
  return pos.reshape(pos.shape[:-1] + (int(pos.shape[-1] / ndim), ndim))

def accumulate_pytree(stack, leaf):
  """
  Accumulate the leafs of a pytree
  """
  if stack is None:
    return leaf
  else:
    return jax.tree_util.tree_map(lambda s, n: s+n, stack, leaf)

def limits(data):
  avg_data = jnp.mean(data)
  std_data = jnp.sqrt(jnp.var(data))
  xmin = avg_data - 4 * std_data
  xmax = avg_data + 4 * std_data
  return (xmin, xmax)

def define_limits(data):
  """
  """
  nwalkers, npart, ndim = data.shape
  q, r, dq, dr = vfeatures(data, npart, ndim)
  lims = {}

  lims['x'] = limits(q[..., 0])
  lims['y'] = limits(q[..., 1])
  lims['z'] = limits(q[..., 2])
  lims['r'] = (0.0, jnp.max(r)*1.025)

  lims['dx'] = limits(dq[..., 0])
  lims['dy'] = limits(dq[..., 1])
  lims['dz'] = limits(dq[..., 2])
  lims['dr'] = (0.0, jnp.max(dr)*1.025)

  return lims

########################################################
########################################################

def basis_change(x, nx, ny, nz):
  """ Computes the new coordinates of a 3D vector in a new
  basis set composed by the orthonormal vector nx, ny, and nz. 
  Args: 
    x: particles positions (np, 3) 
    nx, ny, nz: orthonormal vectors with shape (3)
  Returns:
    rotated particle positions (np, 3) """
  x_new = jnp.dot(x, nx)
  y_new = jnp.dot(x, ny)
  z_new = jnp.dot(x, nz)
  return jnp.array([x_new, y_new, z_new]).T

def features(x, npart, ndim):
  q = jnp.reshape(x, [npart, ndim])
  r = jnp.linalg.norm(q, axis=-1)
  dq = jnp.reshape(q, [1, npart, ndim]) - jnp.reshape(q, [npart, 1, ndim])
  dr = jnp.linalg.norm(dq + jnp.eye(npart)[..., None], axis=-1) * (1 - jnp.eye(npart))
  i, j = jnp.triu_indices(npart, k=1)
  return q, r, dq[i, j, :], dr[i, j]
vfeatures = jax.vmap(features, in_axes=(0, None, None))

def structure_factor_sample(q, k):
  kq = jnp.tensordot(q, k, axes=([1], [0]))
  sk = jnp.mean(jnp.exp(-1j * kq), axis=0)
  return sk
vsf_sample = jax.vmap(structure_factor_sample, in_axes=(0,None))

def structure_factor(q, k):
  sk=jnp.mean(vsf_sample(q, k), axis=0)
  return sk

def reduced_features(pos):
  r = jnp.linalg.norm(pos, axis=-1)
  id1 = jnp.argmin(r)
  q_min = pos[id1, :]
  dq = pos - q_min[None, :]
  dr = jnp.linalg.norm(dq, axis=-1)
  id2 = jnp.argmin(jnp.where(dr != 0.0, dr, jnp.inf))
  nz = jnp.array([0., 0., 1.])
  ny = pos[id2, :] * jnp.array([1., 1., 0.]) / jnp.linalg.norm(pos[id2, :] * jnp.array([1., 1., 0.]), axis=-1)
  nx = jnp.cross(ny, nz)
  v = basis_change(dq, nx, ny, nz)
  q = v # update jax to use delete
#  q = jnp.delete(v, jnp.array([id1, id2]), axis=0, assume_unique_indices=True)
  return q
vreduced_features = jax.vmap(reduced_features)

########################################################
########################################################

def distribution_histogram(data, lims):
  """
  Args:
    data: data to create the histogram in the shape (nsamples, N)
  Returns:
    x: centered bins positions
    y: histogram values
  """

  nsamples, N = data.shape
  data = jnp.reshape(data, [-1,])

  hist, x_ed = jnp.histogram(data, bins=1251, range=lims)
  x = 0.5 * (x_ed[:-1] + x_ed[1:])
  dx = x_ed[1:] - x_ed[:-1]

  norm = 1.0 / (N * nsamples * dx)

  y = hist * norm

  return x, y

def radial_histogram(data, ndim=3, density=True, lims=()):
  """
  Args:
    data: data to create the histogram in the shape (nsamples, N)
  Returns:
    x: centered bins positions
    y: histogram values
  """

  nsamples, N = data.shape
  data = jnp.reshape(data, [-1,])

  hist, r_ed = jnp.histogram(data, bins=1251, range=lims)
  r = 0.5 * (r_ed[:-1] + r_ed[1:])
  dr = r_ed[1:] - r_ed[:-1]

  if ndim == 3:
    norm = 1.0 / (nsamples * 4 * jnp.pi * r**2 * dr)
  elif ndim == 2:
    norm = 1.0 / (nsamples * 2 * jnp.pi * r * dr)
  else:
    print('Error')
    raise SystemExit

  if density:
    norm = norm / N

  y = hist * norm

  return r, y

def coordinates_histograms(data, lims):
  """
  """
  nwalkers, npart, ndim = data.shape

  q, rq, dq, drq = vfeatures(data, npart, ndim)

  x, hx = distribution_histogram(q[...,0], lims['x'])
  y, hy = distribution_histogram(q[...,1], lims['y'])
  z, hz = distribution_histogram(q[...,2], lims['z'])
  r, hr = distribution_histogram(rq, lims['r'])

  dx, hdx = distribution_histogram(dq[...,0], lims['dx'])
  dy, hdy = distribution_histogram(dq[...,1], lims['dy'])
  dz, hdz = distribution_histogram(dq[...,2], lims['dz'])
  dr, hdr = distribution_histogram(drq, lims['dr'])

  rp, hrp = radial_histogram(rq, ndim=3, density=False, lims=lims['r'])
  drp, hdrp = radial_histogram(drq, ndim=3, density=True, lims=lims['dr'])

  pq = jnp.linalg.norm(q[...,:-1], axis=-1)
  dpq = jnp.linalg.norm(dq[...,:-1], axis=-1)
  p, hp = distribution_histogram(pq, lims['r'])
  dp, hdp = distribution_histogram(dpq, lims['dr'])
  pp, hpp = radial_histogram(pq, ndim=2, density=False, lims=lims['r'])
  dpp, hdpp = radial_histogram(dpq, ndim=2, density=True, lims=lims['dr'])

  loc_obs = {}

  loc_obs['x'] = x
  loc_obs['y'] = y
  loc_obs['z'] = z
  loc_obs['r'] = r
  loc_obs['rp'] = rp

  loc_obs['dx'] = dx
  loc_obs['dy'] = dy
  loc_obs['dz'] = dz
  loc_obs['dr'] = dr
  loc_obs['drp'] = drp

  loc_obs['hx'] = hx
  loc_obs['hy'] = hy
  loc_obs['hz'] = hz
  loc_obs['hr'] = hr
  loc_obs['hrp'] = hrp

  loc_obs['hx2'] = hx**2
  loc_obs['hy2'] = hy**2
  loc_obs['hz2'] = hz**2
  loc_obs['hr2'] = hr**2
  loc_obs['hrp2'] = hrp**2

  loc_obs['hdx'] = hdx
  loc_obs['hdy'] = hdy
  loc_obs['hdz'] = hdz
  loc_obs['hdr'] = hdr
  loc_obs['hdrp'] = hdrp

  loc_obs['hdx2'] = hdx**2
  loc_obs['hdy2'] = hdy**2
  loc_obs['hdz2'] = hdz**2
  loc_obs['hdr2'] = hdr**2
  loc_obs['hdrp2'] = hdrp**2
  
  loc_obs['p'] = p
  loc_obs['dp'] = dp
  loc_obs['pp'] = pp
  loc_obs['dpp'] = dpp

  loc_obs['hp'] = hp
  loc_obs['hdp'] = hdp
  loc_obs['hpp'] = hpp
  loc_obs['hdpp'] = hdpp

  loc_obs['hp2'] = hp**2
  loc_obs['hdp2'] = hdp**2
  loc_obs['hpp2'] = hpp**2
  loc_obs['hdpp2'] = hdpp**2

  return loc_obs

########################################################
########################################################

def main():

  dt_sim = {}
  dt_sim['path'] = os.getcwd()

  alpha = 0.1 # feedback parameter for weighted averages
  nk = 251    # number of K by K points in a 2D grid
  Lb = 10.    # limits for the k grid (-Lb,Lb)
  chunk = 64  #

  opt = read_energy_data('train_stats')
  vmc = read_energy_data('vmc_stats')


  ###################################
  # Weighted energy optimization data 
  Wene, Wvar = weighted_averages(opt['energy'], alpha)
  t = opt['step'].values
  ene = Wene[-1]
  std = np.sqrt(Wvar[-1])
  dt_sim['train'] = {'w-avg': [t, Wene, Wvar], 'est': [ene, std]}
  ###################################


  ###################################
  # Reblocking standard error of the mean estimaiton
  bs, ene, std = energy_estimation(vmc)
  dt_sim['vmc'] = {'blocking': [bs, std], 'est': [ene, jnp.max(std)]}
  ###################################


  ###################################
  # Coordinates distributions
  pos = get_samples(path='.', ndim=3)
  pos = pos - pos.mean(axis=-2, keepdims=True) # centrilize for translational invariant systems
  nsteps, nwalkers, npart, ndim = pos.shape
  
  qh = None
  lims = define_limits(pos[0,...])
  for i in range(nsteps):
    obs = coordinates_histograms(pos[i,...], lims)
    qh = accumulate_pytree(qh, obs)
  qh = jax.tree_util.tree_map(lambda n: n/nsteps, qh)

  qh['nsteps'] = nsteps
  ###################################

  ###################################
  # 2D densities
  x = pos[...,0].reshape([-1,])
  y = pos[...,1].reshape([-1,])
  h2d, xed, yed = np.histogram2d(x, y, bins=nk, density=True) 
  n2d = {'xed': xed, 'yed': yed, 'h2d': h2d}

  pos_red = vreduced_features(pos.reshape([-1, npart, ndim]))
  xr = pos_red[...,0].reshape([-1,])
  yr = pos_red[...,1].reshape([-1,])
  h2dr, xred, yred = np.histogram2d(xr, yr, bins=nk, density=True) 
  red_n2d = {'xed': xred, 'yed': yred, 'h2d': h2dr}
  ###################################


  ###################################
  # Defining k points to compute structure factor
  kx = jnp.linspace(-Lb, Lb, nk)
  ky = jnp.linspace(-Lb, Lb, nk)
  kx, ky = jnp.meshgrid(kx, ky)
  kz = jnp.zeros(kx.shape)
  veck = jnp.stack([kx, ky, kz], axis=0)
  
  a_sk = jnp.zeros((nk, nk))
  for i in range(nsteps):
    sk = jnp.zeros((nk, nk))
    for j in range(0, nwalkers, chunk):
      qj = pos[i, j:j+chunk, ...]
      sk += structure_factor(qj, veck)
    a_sk += jnp.abs(sk)**2

  a_sk = jax.tree_util.tree_map(lambda n: n/nsteps, a_sk)
  sk2d = {'kx': kx, 'ky': ky, 'sk': a_sk}
  ###################################


  ###################################
  # Saving all variables in dictionary
  dt_sim['loc-obs'] = {'h': qh, 'sf': sk2d, 'n2d': n2d, 'r-n2d': red_n2d}
  np.save('sim_data.npy', dt_sim)
  ###################################

  return

if __name__ == "__main__":
  main()
