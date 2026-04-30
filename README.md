# SNN Scaling

Status: all simulations running without errors and produce equivalent spike statistics for the same parameter values. The only exception to this is the neuromorphic simulation, which has been implemented but only runs in virtual mode since I do not have access to a board yet.

## Overview

This repo is an experimental framework for studying the computational scaling properties of spiking neural network (SNN) simulations across hardware architectures.

The goal of this project is to systematically characterize how simulation cost depends on:

• Network size (N)

• Connectivity density and mean degree

• Spike statistics and dynamical regime

• Execution model (dense, sparse, event-driven)

• Data representation (dense matrix, sparse matrix, event queue)

• Hardware architecture (CPU, GPU, neuromorphic platforms)

Rather than focusing on task performance or learning, this repository investigates the systems-level behavior of SNN simulation itself.

## Motivation

Spiking neural networks can be simulated using fundamentally different computational paradigms:

• Dense clock-driven matrix operations

• Sparse linear algebra

• Event-driven propagation

• Neuromorphic routing fabrics

Each paradigm interacts differently with hardware architectures such as CPUs, GPUs, and neuromorphic chips.

Understanding when dense simulation is optimal, when sparsity becomes advantageous, when event-driven approaches dominate, and where architectural crossover points occur is critical for developing principled scaling laws and guiding hardware-aware neural modeling.

This project aims to isolate these tradeoffs.

## Setup

Assumes `pyenv` is installed and correctly using version from `.python-version`.

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If running sparse cpu simulation, you will need Intel's mkl library:

```shell
conda install mkl mkl-include
```

Before running `python -m simulations.run_all`, the event-driven C++ backend will need to be compiled with:

```shell
python setup.py build_ext --inplace   
```
