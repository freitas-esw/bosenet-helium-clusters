
"""Evaluating the Hamiltonian on a wavefunction."""

from bosenet import networks
import jax
from jax import lax
import jax.numpy as jnp


def potential_aziz87(dr):
  _eps = 10.948
  _A = 184431.01
  _D = 1.4826
  _alpha = 10.43329537
  _beta = -2.27965105
  _c6 = 1.36745214
  _c8 = 0.42123807
  _c10 = 0.17473318
  f = jnp.where(dr<_D,jnp.exp(-(_D/dr-1.0)**2),1.0)
  return _eps*(_A*jnp.exp(-_alpha*dr+_beta*dr**2)-(_c6/dr**6+_c8/dr**8+_c10/dr**10)*f)


def local_kinetic_energy(f):
  r""" Creates a funciton for the local kinetic energy, -1/2 \nabla^2 ln |f|.

  Args:
  Returns:
  """

  # [hbar^2/(2 m L^2)]/kB for helium 4 and L=2.963 A
  hho2m = 0.69021474872837763

  def _lapl_over_f(params, x):
    n = x.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.grad(f, argnums=1)
    grad_f_closure = lambda y: grad_f(params, y)

    def _body_fun(i, val):
      primal, tangent = jax.jvp(grad_f_closure, (x,), (eye[i],))
      return val + primal[i]**2 + tangent[i]

    return - hho2m * lax.fori_loop(0, n, _body_fun, 0.0)

  return _lapl_over_f


def potential_energy(dr, x_table, y_table):
    """ Returns the potential energy for this particle configuration.

    Args:
    """
    return jnp.interp(dr, x_table, y_table)


def local_energy(f, ndim: int=3):
  """ Creates function to evaluate the local energy.

  Args:
  Returns:
  """
  ke = local_kinetic_energy(f)
  xtable = jnp.linspace(0.50, 5.0, 4000)
  ytable = potential_aziz87(xtable)

  def _e_l(params, x):
    """ Returns the total (local) energy.

    Args:
      params: network parameter.
      x: MCMC configuration.
    """
    _, _, _, r = networks.construct_input_features(x)
    potential = jnp.sum(jnp.triu(potential_energy(r,xtable,ytable), k=1))
    kinetic = ke(params, x)
        
    return potential+kinetic, kinetic, potential

  return _e_l
