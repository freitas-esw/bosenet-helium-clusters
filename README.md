# BNN-HC: Bosonic Neural Networks Helium Clusters 

Bosonic Neural Networks, or Bosenet for short, is an trial wave function based on neural
networks and particullarly adapted for Helium Clusters interacting through the Aziz87
potential[1].

The main results extracted from a version of this algorithm are reported in "Synergy
between deep neural networks and the variational Monte Carlo method for small \(⁴He<sub>N</sub>\)
clusters", William Freitas and S.A.Vitiello, Quantum 7, 1209 (2023).

## Intallation of requiriments

The code was mostly tested using `python3.11`. You also should have installed git.
We recommend the installation of the requiriments inside a python virtual environment.
For more information visit: https://docs.python.org/3/library/venv.html

First, to create the environment use:

```shell
python3.11 -m venv ./venv/bnnhc
```

To activate the environment

```shell
source ./venv/bnnhc/bin/activate
```

The versions specified in the requiriments file are the ones that the tests were performed, change it carefully.
To install the required python libraries, execute:

```shell
pip install -r requiriments 
```

If you have a GPU, and cuda installed, it is recommended to install jaxlib with cuda support. For instance 

```shell
pip install -U "jax[cuda12]"==0.5.0
```

## Usage

The example config file is under the `scripts` directory. To see what kind of parameters you can change,
you should look into the `input.py` file or the `bnnhc/base.py` file. Running the codes looks like:

```shell
cd workspace
python3.11 generate.py
python3.11 ../main.py --config inputs/opt_00.py
python3.11 ../main.py --config inputs/opt_00.py --config.method='vmc'
```

The first line executes the optimisation process, while the second uses the optimised wave function to
compute estimations of the total, kinetic and potential energy. The outputs are the files `train_stats.csv`
and `vmc_stats.csv`.

A simple analysis of the data can be done by executing

```shell
cd he-droplets-n03
python3.11 ../analysis.py
```
The outputs are images in the `.png` format.

## Acknowledgements

The BNN-HC Ansatz is inspired in the FermiNet[2]. 

## Bibliography
[1] A new determination of the ground state interatomic potential for He<sub>2</sub>, Ronald A. Aziz, Frederick R.W. McCourt, and Clement C.K. Wong, Molecular Physics, 1987

[2] FermiNet github, James S. Spencer, David Pfau and FermiNet Contributors, http://github.com/deepmind/ferminet, 2020

## Giving Credit

If you want to use this code or your work is based/inspired by this code, please cite the associated paper mentioned in the beginning.
