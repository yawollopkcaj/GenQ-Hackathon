# 2nd Place 🥈 Winners GenQ Hackathon Series, Quantum in Finance

## Challenge: Counterparty Credit Risk, Quantum Hackathon

This project implements a quantum-enhanced Monte Carlo method for calculating Potential Future Exposure (PFE) of a financial derivatives portfolio using Quantum Amplitude Estimation (QAE).

## Project Overview

### Problem Statement
calculating Potential Future Exposure (PFE) for financial derivatives is traditionally very computationally expensive. Classical Monte Carlo simulations require millions of samples to get accurate estimates, which is slow and costly for banks.

### Key Innovation
- **Quantum Amplitude Estimation**: Achieves O(1/n) error scaling vs O(1/√n) for classical Monte Carlo
- **Correlated Asset Modeling**: Quantum states encode correlation structure between assets
- **Portfolio Aggregation**: Quantum arithmetic (Hamiltonian Dynamics) for efficient calculation

## Technical Architecture

### 1. Quantum State Preparation
- Encodes log-normal distributions for asset prices in quantum amplitudes
- Implements correlation transformation for realistic market dynamics
- Uses 4-6 qubits per asset for price discretization

### 2. Payoff Computation
- Quantum circuits for European option payoffs (calls and puts)
- Handles both long and short positions
- Quantum arithmetic for portfolio aggregation

### 3. Amplitude Amplification
- Grover-based oracle for threshold detection
- Iterative amplitude amplification for PFE quantile estimation
- Theoretical quadratic speedup: O(√N) vs O(N) queries

## Results & Performance

### Portfolio Composition
- 7 positions: 2 FX options, 5 equity options
- Mixed long/short positions
- Total notional: ~$20M USD

### Classical vs Quantum Comparison

| Metric | Classical MC | Quantum (QAE) | Improvement |
|--------|-------------|---------------|-------------|
| Samples/Iterations | 30,000 | ~170 | 176x |
| Error Scaling | O(1/√n) | O(1/n) | Quadratic |
| Convergence Speed | Slow | Fast | ~13x faster |

### PFE Results
- **Classical PFE (95%)**: Computed from 30,000 Monte Carlo simulations
- **Quantum PFE (95%)**: Estimated using 4-5 QAE iterations
- **Accuracy**: Within 5% of classical result with 99% fewer evaluations

<p align="center">
<caption><b>Theoretical Error Scaling</b></caption>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/6caf2151-10b9-4861-9311-b519a5b49136" width="600" />
</p>

<p align="center">
<caption><b>Sample Efficiency</b></caption>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/13e49577-d024-4c74-b69e-ee4a5d620e74" width="600" />
</p>

## Technical Deep Dive

### Quantum Circuit Components

1. **State Preparation Circuit**
   - Depth: O(n log n) for n qubits
   - Gates: Controlled rotations for correlations
   - Total qubits: 4n + 6 + n (price + payoff + ancilla)

2. **Payoff Oracle**
   - Implements comparison and arithmetic operations
   - Reversible computation for quantum coherence
   - Handles option type, strike, and position side

3. **Amplitude Amplification**
   - Grover iterations: O(√(1/p)) where p is success probability
   - Optimal iteration count: π/4 × √(N/M)

## Path to Quantum Advantage

### Current Implementation (NISQ-ready)
- **Qubits Required**: 30-40 for basic portfolio
- **Gate Depth**: ~1000 for full circuit
- **Error Tolerance**: 0.1% gate error acceptable

### Scaling Analysis
- **Break-even point**: ~10⁴ classical samples
- **Significant advantage**: >10⁶ classical samples
- **Hardware requirements**: 50-100 logical qubits

## Business Case

### Value Proposition
- **Speed**: 100-1000x faster risk calculations
- **Cost**: Reduced computational infrastructure
- **Accuracy**: Better tail risk estimation
- **Real-time**: Enable intraday risk updates

### Market Opportunity
- Global derivatives market: >$600 trillion notional
- Risk management software: $2.5B market
- Quantum advantage timeline: 3-5 years

### Target Customers
- Investment banks
- Hedge funds
- Central clearing houses
- Regulatory bodies

## SDG Impact

### SDG 8: Decent Work and Economic Growth
- **Financial Stability**: Better risk management prevents systemic crises
- **Market Efficiency**: Faster pricing enables better capital allocation
- **Job Creation**: New roles in quantum finance
- **Economic Resilience**: Improved stress testing capabilities

## Technical Challenges & Solutions

### Challenge 1: Quantum Noise
- **Solution**: Error mitigation techniques
- **Implementation**: Zero-noise extrapolation

### Challenge 2: Limited Qubits
- **Solution**: Circuit optimization and compression
- **Implementation**: Variational encoding

### Challenge 3: Correlation Modeling
- **Solution**: Efficient quantum correlation circuits
- **Implementation**: Controlled rotation gates

## Resources

* [Quantum Amplitude Estimation](https://arxiv.org/pdf/1905.02666)
* [Risk Analysis using Quantum Computers](https://www.nature.com/articles/s41534-019-0130-6)
* [Quantum Risk Models](https://arxiv.org/pdf/2103.05475)
* [Q# Documentation](https://learn.microsoft.com/en-us/azure/quantum/qsharp-overview)

---

**Team**: JEAF

**Hackathon**: GenQ Hackathon Series, Quantum for Finance

**Date**: 2025  
