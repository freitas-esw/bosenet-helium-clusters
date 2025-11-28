
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex

import os 
import pandas as pd
import math as mt
import numpy as np
import jax.numpy as jnp

########################################################
########################################################

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Set plot config - APS style
plt.rcParams['figure.dpi']     = 300
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

def result_str(value, error):
  d = - int(mt.floor(mt.log10(jnp.abs(error))))
  e = int(jnp.round(error, d) * 10**d)
  v = jnp.round(value , d)
  return f'{v:.{d}f}'+'({:n})'.format(e)

def adjust_color(color, factor):
  c = np.array(to_rgb(color))
  if factor > 0:
    # lighten: blend toward white
    c = c + (1 - c) * factor
  else:
    # darken: blend toward black
    c = c * (1 + factor)
  return to_hex(c.clip(0, 1))

def errorbar_plot(xs, ys, vs, n, xlabel, ylabel, figname):
  x = xs
  y = ys
  e = jnp.sqrt((vs-ys**2)/n)
  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  ax.errorbar(x, y, yerr=e, marker='.', c=colors[0], ms=1, ls=' ', elinewidth=1, ecolor=adjust_color(colors[0], 0.6))
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  dy = jnp.max(y) - jnp.min(y)
  ax.set_ylim([jnp.min(y) - dy * 0.05, jnp.max(y) + dy * 0.125])
  ax.set_box_aspect(0.95)
  fig.tight_layout()
  fig.savefig(figname, transparent=True)
  plt.close(fig)
  return

########################################################
########################################################

def main():

  img_fmt = '.png'

  smin = 10
  smax = 20

  opt = read_energy_data('train_stats')
  vmc = read_energy_data('vmc_stats')


  t = opt['step'].values
  y = opt['energy'].values
  
  i = int(t.size/10) + 1
  yb = jnp.mean(y[-i:])
  dy = jnp.sqrt(jnp.var(y[-i:])) 
   
  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  ax.scatter(t, y)
  ax.set_ylim([yb - smin*dy, yb + smax*dy])
  ax.set_ylabel('Energy [$E_0$]') 
  ax.set_xlabel('Optimization steps')
  fig.tight_layout()
  fig.savefig('opt_ene' + img_fmt, transparent=True)
  plt.close(fig)
 

  t = vmc['step'].values
  y = vmc['energy'].values

  dy = 4. * jnp.sqrt(jnp.var(y))
  lims = (y.mean() - dy, y.mean() + dy)

  hist, bins = jnp.histogram(y, bins = int(jnp.sqrt(t[-1])), range=lims)

  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  ax.stairs(hist, bins)
  ax.set_ylabel('Counts')
  ax.set_xlabel('Energy [$E_0$]') 
  fig.tight_layout()
  fig.savefig('ene_hist' + img_fmt, transparent=True)
  plt.close(fig)
   
  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  ax.scatter(t, y)
  ax.set_ylabel('Energy [$E_0$]') 
  ax.set_xlabel('Iteration')
  fig.tight_layout()
  fig.savefig('ene_iter' + img_fmt, transparent=True)
  plt.close(fig)

      
  data = np.load('sim_data.npy', allow_pickle=True).tolist()

  t, y, v = data['train']['w-avg']
  s = int(t[-1]/180)
  
  ene, std = data['train']['est']
  ene_est = result_str(ene, std) if std < 10. else str(ene)
  
  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  ax.errorbar(t[::s], y[::s], yerr=np.sqrt(v[::s]), marker='.', ms=4, ls='', elinewidth=1)
  ax.set_ylim([ene - smin*std, ene + smax*std])
  ax.set_box_aspect(0.9)
  fig.text(0.5, 0.7, '$E=$'+ene_est+'$[E_0]$', size='small')
  ax.set_xlabel('Optimization steps')
  ax.set_ylabel('Energy [$E_0$]')
  fig.tight_layout()
  fig.savefig('wene_opt' + img_fmt, transparent=True)
  plt.close(fig)


  t, y = data['vmc']['blocking']
  
  ene, std = data['vmc']['est']
  ene_est = result_str(ene, std) if std < 10. else str(ene)

  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  ax.scatter(t, y)
  ax.set_ylabel('SE Energy [$E_0$]') 
  ax.set_xlabel('Block size')
  fig.text(0.65, 0.25, '$E=$'+ene_est+'$[E_0]$', size='small')
  fig.tight_layout()
  fig.savefig('sem_ene' + img_fmt, transparent=True)
  plt.close(fig)


  qh = data['loc-obs']['h']
  n = qh['nsteps'] - 1
  
  errorbar_plot(qh['x'],   qh['hx'],   qh['hx2'],   n, xlabel='$x_{i} [R_0]$',  ylabel='$P(x_{i})$ [$R_0^{-1}$]',  figname='px-distribution'  + img_fmt)
  errorbar_plot(qh['y'],   qh['hy'],   qh['hy2'],   n, xlabel='$y_{i} [R_0]$',  ylabel='$P(y_{i})$ [$R_0^{-1}$]',  figname='py-distribution'  + img_fmt)
  errorbar_plot(qh['z'],   qh['hz'],   qh['hz2'],   n, xlabel='$z_{i} [R_0]$',  ylabel='$P(z_{i})$ [$R_0^{-1}$]',  figname='pz-distribution'  + img_fmt)
  errorbar_plot(qh['r'],   qh['hr'],   qh['hr2'],   n, xlabel='$r_{i} [R_0]$',  ylabel='$P(r_{i})$ [$R_0^{-1}$]',  figname='pr-distribution'  + img_fmt)
  errorbar_plot(qh['dx'],  qh['hdx'],  qh['hdx2'],  n, xlabel='$x_{ij} [R_0]$', ylabel='$P(x_{ij})$ [$R_0^{-1}$]', figname='pdx-distribution' + img_fmt)
  errorbar_plot(qh['dy'],  qh['hdy'],  qh['hdy2'],  n, xlabel='$y_{ij} [R_0]$', ylabel='$P(y_{ij})$ [$R_0^{-1}$]', figname='pdy-distribution' + img_fmt)
  errorbar_plot(qh['dz'],  qh['hdz'],  qh['hdz2'],  n, xlabel='$z_{ij} [R_0]$', ylabel='$P(z_{ij})$ [$R_0^{-1}$]', figname='pdz-distribution' + img_fmt)
  errorbar_plot(qh['dr'],  qh['hdr'],  qh['hdr2'],  n, xlabel='$r_{ij} [R_0]$', ylabel='$P(r_{ij})$ [$R_0^{-1}$]', figname='pdr-distribution' + img_fmt)
  errorbar_plot(qh['rp'],  qh['hrp'],  qh['hrp2'],  n, xlabel='$r_{i} [R_0]$',  ylabel='$n(r_{i})$ [$R_0^{-3}$]',  figname='pr-correlation'   + img_fmt)
  errorbar_plot(qh['drp'], qh['hdrp'], qh['hdrp2'], n, xlabel='$r_{ij} [R_0]$', ylabel='$g(r_{ij})$ [$R_0^{-3}$]', figname='pdr-correlation'  + img_fmt)

  errorbar_plot(qh['p'],   qh['hp'],   qh['hp2'],   n, xlabel='$\\rho_{i} [R_0]$',  ylabel='$P(\\rho_{i})$ [$R_0^{-1}$]',  figname='pp-distribution' +img_fmt)
  errorbar_plot(qh['dp'],  qh['hdp'],  qh['hdp2'],  n, xlabel='$\\rho_{ij} [R_0]$', ylabel='$P(\\rho_{ij})$ [$R_0^{-1}$]', figname='pdp-distribution'+img_fmt)
  errorbar_plot(qh['pp'],  qh['hpp'],  qh['hpp2'],  n, xlabel='$\\rho_{i} [R_0]$',  ylabel='$n(\\rho_{i})$ [$R_0^{-2}$]',  figname='pp-correlation'  +img_fmt)
  errorbar_plot(qh['dpp'], qh['hdpp'], qh['hdpp2'], n, xlabel='$\\rho_{ij} [R_0]$', ylabel='$g(\\rho_{ij})$ [$R_0^{-2}$]', figname='pdp-correlation' +img_fmt)


  kx = data['loc-obs']['sf']['kx']
  ky = data['loc-obs']['sf']['ky']
  sk = data['loc-obs']['sf']['sk']
  
  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  pcm = ax.pcolormesh(kx, ky, sk, shading='auto', cmap='viridis', rasterized=True)
  fig.colorbar(pcm, ax=ax, label='S($\\mathbf {k}$)')
  ax.set_xlabel('$k_x$ [$R_0^{-1}$]')
  ax.set_ylabel('$k_y$ [$R_0^{-1}$]')
  ax.set_box_aspect(1.0)
  fig.tight_layout()
  fig.savefig('sk_cmap' + img_fmt, transparent=True)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  pcm = ax.pcolormesh(kx, ky, jnp.log(sk), shading='auto', cmap='viridis', rasterized=True)
  fig.colorbar(pcm, ax=ax, label='$\\ln[S(\\mathbf {k}$)]')
  ax.set_xlabel('$k_x$ [$R_0^{-1}$]')
  ax.set_ylabel('$k_y$ [$R_0^{-1}$]')
  ax.set_box_aspect(1.0)
  fig.tight_layout()
  fig.savefig('logsk_cmap' + img_fmt, transparent=True)
  plt.close(fig)


  n2d = data['loc-obs']['n2d']['h2d']
  xed = data['loc-obs']['n2d']['xed']
  yed = data['loc-obs']['n2d']['yed']

  L = 8.1
  
  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  im = ax.imshow(n2d.T, origin='lower', extent=[-L, L, -L, L], cmap='inferno', aspect='auto')
  ax.set_xlabel('$x$ [$R_0$]')
  ax.set_ylabel('$y$ [$R_0$]')
  fig.colorbar(im, label='$n(x,y)$')
  ax.set_box_aspect(1.0)
  fig.tight_layout()
  fig.savefig('n2d_cmap' + img_fmt, transparent=True)
  plt.close(fig)
  
  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  im = ax.imshow(jnp.log(n2d.T), origin='lower', extent=[-L, L, -L, L], cmap='inferno', aspect='auto')
  ax.set_xlabel('$x$ [$R_0$]')
  ax.set_ylabel('$y$ [$R_0$]')
  fig.colorbar(im, label='$\\ln[n(x,y)]$')
  ax.set_box_aspect(1.0)
  fig.tight_layout()
  fig.savefig('logn2d_cmap' + img_fmt, transparent=True)
  plt.close(fig)

  n2d = data['loc-obs']['r-n2d']['h2d']
  xed = data['loc-obs']['r-n2d']['xed']
  yed = data['loc-obs']['r-n2d']['yed']
  
  fig, ax = plt.subplots(figsize=[5.5, 4.5], dpi=300)
  im = ax.imshow(n2d.T, origin='lower', extent=[xed[0], xed[-1], yed[0], yed[-1]], cmap='inferno', aspect='auto')
  ax.set_xlabel('$x$ [$R_0$]')
  ax.set_ylabel('$y$ [$R_0$]')
  fig.colorbar(im, label='$n_{\\rm {r}}(x,y)$')
  ax.set_box_aspect(1.0)
  fig.tight_layout()
  fig.savefig('reduced_n2d_cmap'+img_fmt, transparent=True)
  plt.close(fig)
 
  return

if __name__ == "__main__":
  main()
