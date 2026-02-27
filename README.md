# SNN Scaling

## Overview

This repo is an experimental framework for studying the computational scaling properties of spiking neural network (SNN) simulations across graph structures and hardware architectures.

The goal of this project is to systematically characterize how simulation cost depends on:

• Network size (N)

• Connectivity density and mean degree

• Graph topology (random, small-world, scale-free, modular, etc.)

• Spike statistics and dynamical regime

• Execution model (dense, sparse, event-driven)

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

## Project Scope (Initial Phase)

The first milestone focuses on dense, clock-driven SNN simulation. Implementation will be on PyTorch comparing CPU vs GPU runtimes.

Future phases will introduce:

• Sparse simulation backends

• Event-driven implementations

• Structured graph generators

• Energy and runtime measurements

• Neuromorphic software stack comparisons

• Extraction of scaling laws

## Research Utility

This framework is intended to:

• Provide controlled benchmarking for SNN simulation

• Identify computational crossover regimes

• Inform architecture-aware neural modeling

• Support reproducible scaling studies

The long-term objective is to characterize how graph structure, sparsity, and spike statistics interact with hardware architecture to determine simulation efficiency.

## Status

Early-stage development, beginning with dense PyTorch baselines.