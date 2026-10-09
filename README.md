# bio-lif-core

LIF neuron with exact integrator — the core of the BioMorph project.

## Overview

This package implements a Leaky Integrate-and-Fire (LIF) neuron with:

- Physical units: mV, nA, MΩ, ms (1 MΩ · 1 nA = 1 mV).
- Exact analytical integrator: unconditionally stable for any dt > 0.
- Strong validation: all parameters checked for finiteness and physical constraints.
- Strong exception guarantee: validation errors never modify the neuron state.

The neuron does NOT store spike history — this is the responsibility of a separate recorder module.

## Mathematical model

The LIF equation:

    tau_m * dV/dt = -(V - V_rest) + R_m * I

Exact solution for constant current over step dt:

    V_inf = V_rest + R_m * I
    V(t+dt) = V_inf + (V(t) - V_inf) * exp(-dt / tau_m)

## Installation

    pip install -e ".[test]"

## Quick start

    from biomorph import LIFNeuron

    n = LIFNeuron()
    spike = n.step(input_current=0.2, t=0.0, dt=0.1)
    print(n)

## Testing

    pytest tests/ -v

98 unit tests cover:

- Physics: sub-threshold, threshold crossing, refractory period.
- Numerical accuracy: exact integration, composition property, dt independence.
- Corner cases: NaN, inf, overflow, dt from 0.001 to 1000 ms.
- Exception safety: validation errors do not modify state.

## Status

- [x] LIFNeuron v0.1 — scalar reference implementation
- [ ] LIFNetwork — vectorized NumPy implementation
- [ ] synapse — sparse synaptic connections
- [ ] recorder — spike event recording
- [ ] Reference comparison with Brian2 (Shiu et al. 2024)

## Project context

This repository is part of BioMorph — a research project on biomorphic systems based on the Drosophila melanogaster connectome.

The long-term goal: experimentally test whether biological topology (connectome-derived SNN) provides measurable computational advantages over random graphs and learned baselines.

## License

MIT