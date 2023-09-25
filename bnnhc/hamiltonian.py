
"""Evaluating the Hamiltonian on a wavefunction."""

from bnnhc import networks
import jax
from jax import lax
import jax.numpy as jnp

def potential_lj612(dr):
  """
  Computes the Lennard-Jones potential

  Args: 
    dr: relative distance between atoms
  Returns:
    pot: interaction potential value in Kelvin
  """
  # Potential parameters
  _eps = 4.*10.22        
  return _eps*( 1./dr**12 - 1./dr**6 )


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


def local_kinetic_energy(f, lmb):
  r""" Creates a funciton for the local kinetic energy, -lmb/2 \nabla^2 ln |f|.

  Args:
    f: function that evaluates the logarithmic wavefunction
    lmb: hbar^2 / ( 2 m l^2 k_B ) value in Kelvin

  Returns: function that evaluates the local kinetic energy
           -lmb/2 \nabla^2 ln |f|
  """

  def _lapl_over_f(params, x):
    n = x.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.grad(f, argnums=1)
    grad_f_closure = lambda y: grad_f(params, y)

    def _body_fun(i, val):
      primal, tangent = jax.jvp(grad_f_closure, (x,), (eye[i],))
      return val + primal[i]**2 + tangent[i]

    return - lmb * lax.fori_loop(0, n, _body_fun, 0.0)

  return _lapl_over_f


def local_energy(f, pot_type: str='aziz'):
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

  # Kinetic energy pre-factor in Kelvin
  # [hbar^2/(2 m L^2)]/kB = 0.69021474872837763 for helium 4 mass and L=2.963 A
  # [hbar^2/(2 m L^2)]/kB = 0.927525459         for helium 4 mass and L=2.556 A  
  potential_energy = potential_aziz87 if pot_type == 'aziz' else potential_lj612
  hho2m = 0.69021474872837763 if pot_type == 'aziz' else 0.927525459
  
  ke = local_kinetic_energy(f, hho2m)

  def _e_l(params, x):
    """ Computes the total local energy, potential and kinetic energy.

    Args:
      params: network parameters
      x: particles configuration set

    Returns:
      local energy, kinetic energy, potential energy
    """
    _, _, _, r = networks.construct_input_features(x)
    potential = jnp.sum(jnp.triu(potential_energy(r), k=1))
    kinetic = ke(params, x)
        
    return potential+kinetic, kinetic, potential

  return _e_l
