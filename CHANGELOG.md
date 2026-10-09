# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- `LIFNetwork` — vectorized NumPy implementation with sparse CSR support
- `synapse` — sparse synaptic connections (models A, B, C)
- `recorder` — spike event recording
- Reference comparison with Brian2 (Shiu et al. 2024)

## [0.1.0] — 2026-10-08

### Added
- `LIFNeuron` — Leaky Integrate-and-Fire neuron with physical units (mV, nA, MΩ, ms)
- Exact analytical integrator, unconditionally stable for any `dt > 0`
- Strong validation: finiteness and physical constraints for all parameters
- Strong exception guarantee: validation errors never modify neuron state
- 98 unit tests covering physics, numerical accuracy, corner cases, and exception safety
- `pyproject.toml` with `requires-python >= 3.10`, numpy dependency, test extras
- `README.md` with overview, mathematical model, installation, quick start

### Documentation
- Comprehensive README with LIF equation and exact solution
- MIT license

[Unreleased]: https://github.com/timur-akhmadeev/bio-lif-core/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/timur-akhmadeev/bio-lif-core/releases/tag/v0.1.0
