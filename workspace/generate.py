import os
import itertools 

npart = [3, 5, 10]
strength = [0,]

def generate_opt_input(npart):
  string = f"""from src import base
def get_config():
  cfg = base.default()
  cfg.method = 'train'
  cfg.debug.deterministic = True
  cfg.debug.seed = 42
  cfg.batch_size = 4096
  cfg.system.np = {npart}
  cfg.optim.iterations = 20000 
  cfg.optim.lr.rate = 0.01
  cfg.mcmc.init_width = 4.
  cfg.mcmc.width = 0.01
  cfg.mcmc.burn_in = 0
  cfg.log.save_frequency = 5000
  cfg.log.save_path = './he-droplets-n{npart:02d}/'
  cfg.log.restore_path = './he-droplets-n{npart:02d}/'
  cfg.network.hidden_dims = ((128,8),(128,8),(128,8),(128,8))
  cfg.network.orbitals = 4
  cfg.debug.check_nan = False
  return cfg
"""
  return string

def generate_vmc_input(npart):
  string = f"""from src import base
def get_config():
  cfg = base.default()
  cfg.method = 'vmc'
  cfg.debug.deterministic = True
  cfg.debug.seed = 42
  cfg.batch_size = 4096
  cfg.system.np = {npart}
  cfg.mcmc.init_width = 4.
  cfg.mcmc.width = 0.01
  cfg.mcmc.burn_in = 0
  cfg.log.save_path = './he-droplets-n{npart:02d}/'
  cfg.log.restore_path = './he-droplets-n{npart:02d}/'
  cfg.network.hidden_dims = ((128,8),(128,8),(128,8),(128,8))
  cfg.network.orbitals = 4
  cfg.vmc.block_size = 4096
  cfg.vmc.iterations = 4000
  cfg.debug.check_nan = False
  return cfg
"""
  return string

if __name__ == "__main__":

  k = 0
  for n, w in itertools.product(npart, strength):
    opt_in = generate_opt_input(n)
    vmc_in = generate_vmc_input(n)
    if not os.path.exists('inputs'):
      os.makedirs('inputs')
    f = open(f'inputs/opt_{k:02d}.py', 'w')
    f.write(opt_in)
    f.close()
    f = open(f'inputs/vmc_{k:02d}.py', 'w')
    f.write(vmc_in)
    f.close()
    k+=1

