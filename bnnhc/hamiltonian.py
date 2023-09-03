
"""Evaluating the Hamiltonian on a wavefunction."""

from bhc import networks
import jax
from jax import lax
import jax.numpy as jnp


def potential_aziz87(dr):
  """ 
  Computes the HFDB(HE) potential from Aziz et.al. 1987. https://doi.org/10.1080/00268978700101941

  Args: 
    dr: relative distance between atoms

  Returns:
    pot: interaction potential value in Kelvin
  """
  
  # Potential parameters
  _eps = 10.948
  _A = 184431.01
  _D = 1.4826
  _alpha = 10.43329537
  _beta = -2.27965105
  _c6 = 1.36745214
  _c8 = 0.42123807
  _c10 = 0.17473318

  # Numerical computation of the interaction
  f = jnp.where(dr<_D,jnp.exp(-(_D/dr-1.0)**2),1.0)
  pot = _eps*(_A*jnp.exp(-_alpha*dr+_beta*dr**2)-(_c6/dr**6+_c8/dr**8+_c10/dr**10)*f)

  return pot


def local_kinetic_energy(f):
  r""" Creates a funciton for the local kinetic energy, -1/2 \nabla^2 ln |f|.

  Args:
    f: function that evaluates the logarithmic wavefunction

  Returns: function that evaluates the local kinetic energy
           -1/2 \nabla^2 ln |f|
  """

  # Kinetic energy pre-factor in Kelvin
  # [hbar^2/(2 m L^2)]/kB for helium 4 mass and L=2.963 A
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
  """ Computes the potential energy for a pair distance given a table.

  Args:
    dr: relative distance for a pair of particles
    x_table: table with values of distances 
    y_table: table with potential values

  Returns: Interpolation of the table for the dr value 
  """
  return jnp.interp(dr, x_table, y_table)


def local_energy(f, ndim: int=3):
  """ Construct the table for the potential and creates a function 
      to evaluate the local energy.

  Args:
    f: function that returns the log of the wavefunction given the
       variational parameters and a configuration state
    ndim: number of spatial dimenssions

  Returns: function that evaluates the local total energy given a
           set of variational parameters params and a single set 
           of particle positions
  """
  ke = local_kinetic_energy(f)
  xtable = jnp.linspace(0.50, 5.0, 4000)
  ytable = potential_aziz87(xtable)

  def _e_l(params, x):
    """ Computes the total local energy, potential and kinetic energy.

    Args:
      params: network parameters
      x: particles configuration set

    Returns:
      local energy, kinetic energy, potential energy
    """
    _, _, _, r = networks.construct_input_features(x)
    potential = jnp.sum(jnp.triu(potential_energy(r,xtable,ytable), k=1))
    kinetic = ke(params, x)
        
    return potential+kinetic, kinetic, potential

  return _e_l
