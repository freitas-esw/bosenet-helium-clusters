import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as mt

# Define function that returns a string in the format:
# Real value +- Real error
def result_str( value, error ):
  d = - int( mt.floor( mt.log10( abs(error) ) ) )
  e = round( error, d )
  v = round( value , d )
  return f' {v}'+'+-'+'{:n}; '.format(e)
####

# Define function that returns a string in the format:
# Real value (Int error)
def result_str2( value, error ):
  d = - int( mt.floor( mt.log10( abs(error) ) ) )
  e = int(round( error, d )*10**d)
  v = round( value , d )
  return f' {v}'+'({:n}); '.format(e)
####

####
def mean_std_i( energy_array, i ):
  if ( i==0 ):
    return np.array( [energy_array[i], np.nan] )
  else:
    return np.array( [np.mean(energy_array[:i+1]), block_std_dev( energy_array[:i+1] )] )
####

####
def block_std_dev( vec ):
  mean = np.mean(vec)
  ms2  = mean**2
  vs2  = vec**2
  std  = np.sqrt( (np.mean(vs2)-ms2) / ( np.size(vec)-1 ) )
  return std
####

########################################################
########################################################

def set_params_plt():

  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color']

# Set plot config - APS style
  plt.rcParams['figure.dpi']     = 100
  plt.rcParams['font.size']      = 17
  plt.rcParams['axes.linewidth'] = 1.25

  plt.rcParams['font.weight']        = 'normal'
  plt.rcParams['axes.labelweight']   = 'normal'
  plt.rcParams['axes.titleweight']   = 'normal'

  plt.rcParams['legend.frameon'] = False
  plt.rcParams['legend.loc']     = 'upper center'

  plt.rcParams['xtick.labelsize']     = 15
  plt.rcParams['xtick.direction']     = 'in'
  plt.rcParams['xtick.major.width']   = 1.25
  plt.rcParams['xtick.major.size']    = 5
  plt.rcParams['xtick.minor.visible'] = True
  plt.rcParams['xtick.minor.width']   = 1.25
  plt.rcParams['xtick.minor.size']    = 3.5
  plt.rcParams['xtick.top']           = 'on'

  plt.rcParams['ytick.labelsize']     = 15
  plt.rcParams['ytick.direction']     = 'in'
  plt.rcParams['ytick.major.width']   = 1.25
  plt.rcParams['ytick.major.size']    = 5
  plt.rcParams['ytick.minor.visible'] = True
  plt.rcParams['ytick.minor.width']   = 1.25
  plt.rcParams['ytick.minor.size']    = 3.5
  plt.rcParams['ytick.right']         = 'on'

  plt.rcParams['text.usetex'] = False
  plt.rcParams['mathtext.fontset'] = 'cm'
  plt.rcParams['font.family'] = 'STIXGeneral'

  return colors

########################################################
########################################################

def main():

  colors = set_params_plt()

  path = 'train_stats.csv'
  data = pd.read_csv(path, header=None, skiprows=1).values

  fig, ax = plt.subplots( figsize=[5,4] )
  ax.scatter( data[:,0], data[:,1], marker='.', s=5, c=colors[0] )
  ax.set_ylabel('Energy [K]')
  ax.set_xlabel('Optimisation iterations')
  if len(data[0,:]) > 1e5:
    ax.set_xscale('log')
  fig.tight_layout()
  fig.savefig( 'optimisation.png', dpi=200 )


  path = 'vmc_stats.csv'
  data = pd.read_csv(path, header=None, skiprows=1).values
  j = len( data[:,0] )//2
  
  ene, std = mean_std_i( data[j:,1], np.size(data[j:,1]) )
  pot, ept = mean_std_i( data[j:,4], np.size(data[j:,4]) )
  kin, ekn = mean_std_i( data[j:,3], np.size(data[j:,3]) )

  f = open( 'estimations.out', 'w' )
  f.write( ' Energy; EneErr; Kinetic; KinErr; Potential; PotErr  \n' )
  f.write( ' {}; {}; {}; {}; {}; {} \n'.format(ene, std, kin, ekn, pot, ept) )
#  f.write( result_str(ene,std) +  result_str(kin,ekn) +  result_str(pot,ept) + '\n'  )
#  f.write( result_str2(ene,std) + result_str2(kin,ekn) + result_str2(pot,ept) + '\n' )
  return

if __name__ == "__main__":
  main()
