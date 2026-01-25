# 2nd Place 🥈 Winners GenQ Hackathon Series, Quantum in Finance

## Challenge: Counterparty Credit Risk, Quantum Hackathon

### Summary 

This project implements a quantum-enhanced Monte Carlo method for calculating Potential Future Exposure (PFE) of a financial derivatives portfolio using Quantum Amplitude Estimation (QAE).

## Project Overview

### Problem Statement
calculating Potential Future Exposure (PFE) for financial derivatives is traditionally very computationally expensive. Classical Monte Carlo simulations require millions of samples to get accurate estimates, which is slow and costly for banks.

<caption><i>Potential Future Exposure (PFE)</i></caption>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/df7346b5-7df2-4a11-8ca6-1c94e0a6e888" width="600" />
</p>

## Technical Architecture

### Architectural Overview
This system utilizes **Quantum Amplitude Estimation (QAE)** to evaluate financial risk metrics (such as Value at Risk or Conditional Value at Risk) and price European options. The architecture provides a theoretical quadratic speedup over classical Monte Carlo methods by leveraging quantum interference to converge on the target value with fewer query samples.

<p align="center">
<caption><i>Quantum Amplitude Estimation</i></caption>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/f8b23ed4-dcdf-4a85-8410-bd5475132564" width="600" />
</p>

### 1. Quantum State Preparation (Data Encoding)
The initialization phase loads classical financial probability distributions into a quantum superposition state.

* **Distribution Encoding:** Maps log-normal distributions of asset prices into the amplitudes of quantum states.
* **Correlation Structure:** Implements multivariate dependence between assets using a series of controlled rotation gates ($R_y, R_z$).
* **Discretization:** Utilizes **4–6 qubits per asset** to represent price intervals, balancing precision with circuit depth.
* **Circuit Depth:** Scales as $O(n \log n)$ for $n$ qubits, optimizing for coherence time constraints.

### 2. Payoff Operator & Portfolio Aggregation
Once the state is prepared, the system computes the financial logic using reversible quantum arithmetic.

* **Payoff Logic:** Dedicated quantum comparators calculate the payoff for European options (calls and puts), handling both long and short positions efficiently.
* **Portfolio Aggregation:** Uses quantum adders to sum individual asset payoffs into a total portfolio value. This preserves the quantum coherence required for subsequent amplification.
* **Resource Overhead:** Requires approximately $4n + 6$ qubits (encoding + payoff) plus additional ancilla qubits for arithmetic carry operations.

### 3. Amplitude Estimation Engine (The Solver)
The core computational engine replaces classical sampling with Grover-based interference patterns to estimate the expectation value.

* **Oracle Construction:** A Boolean oracle identifies states where the portfolio loss exceeds a specific threshold (for VaR calculations).
* **Grover Operator:** Iteratively amplifies the amplitude of the "risk" states. The optimal number of iterations is $\frac{\pi}{4}\sqrt{N/M}$, where $N$ is the total search space and $M$ is the number of target states.
* **Convergence Rate:** Achieves a convergence error scaling of $O(1/n)$ (Heisenberg limit) compared to the classical Monte Carlo scaling of $O(1/\sqrt{n})$.
  
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

## Logistics and Constraints

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

## References

1. N. Stamatopoulos, D. J. Egger, Y. Sun, C. Zoufal, R. Iten, N. Shen, and S. Woerner, "Option Pricing using Quantum Computers," Quantum, vol. 4, p. 291, Jul. 2020. [Online]. Available: https://arxiv.org/pdf/1905.02666

2. S. Woerner and D. J. Egger, "Quantum risk analysis," npj Quantum Information, vol. 5, no. 15, Feb. 2019. [Online]. Available: https://www.nature.com/articles/s41534-019-0130-6

3. M. C. Braun, T. Decker, N. Hegemann, S. F. Kerstan, and C. Schäfer, "A Quantum Algorithm for the Sensitivity Analysis of Business Risks," arXiv preprint arXiv:2103.05475, Mar. 2021. [Online]. Available: https://arxiv.org/pdf/2103.05475

4. Microsoft, "Introduction to the Q# Programming Language," Microsoft Azure Quantum Documentation. [Online]. Available: https://learn.microsoft.com/en-us/azure/quantum/qsharp-overview

---

**Team Name**: JEAF (**J**ack, **E**d, **A**lissa, and **F**araz)

**Hackathon**: GenQ Hackathon Series, Quantum AI for Finance (Singapore)

**Date**: 2025  
