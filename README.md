# BNN-HC: Bosonic Neural Networks Helium Clusters 

Bosonic Neural Networks, or Bosenet for short, is an trial wave function based on neural
networks and particullarly adapted for Helium Clusters interacting through the Aziz87
potential[1].

The main results extracted from this version of the algorithm are reported in "Synergy
between deep neural networks and the variational Monte Carlo method for small \(⁴He<sub>N</sub>\)
clusters", William Freitas and S.A.Vitiello, arXiv:2302.00599

## Intallation of requiriments

The code was mostly tested using `python3.10` and `python3.8`. You also should have installed git.
We recommend the installation of the requiriments inside a python virtual environment.
For more information visit: https://docs.python.org/3/library/venv.html

First, to create the environment use:

```shell
python3.10 -m venv ./venv/bnnhc
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
pip install jaxlib==0.1.75+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Usage

The example config file is under the `scripts` directory. To see what kind of parameters you can change,
you should look into the `input.py` file or the `bnnhc/base_config.py` file. Running the codes looks like:

```shell
python3.10 bose.py --config scripts/he02n/input.py
python3.10 vmcbose.py --config scripts/he02n/input.py
```

The first line executes the optimisation process, while the second uses the optimised wave function to
compute estimations of the total, kinetic and potential energy. The outputs are the files `train_stats.csv`
and `vmc_stats.csv`.

A simple analysis of the data can be done by executing

```shell
cd scripts/he02n/
python3.10 ../analysis.py
```
The outputs are an image called `optimisation.png` and a text file `estimations.out`

## Acknowledgements

The BNN-HC Ansatz is inspired in the FermiNet[2]. 

## Bibliography
[1] A new determination of the ground state interatomic potential for He<sub>2</sub>, Ronald A. Aziz, Frederick R.W. McCourt, and Clement C.K. Wong, Molecular Physics, 1987

[2] FermiNet github, James S. Spencer, David Pfau and FermiNet Contributors, http://github.com/deepmind/ferminet, 2020

