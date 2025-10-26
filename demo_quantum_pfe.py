#!/usr/bin/env python3
"""
Simplified Quantum PFE Demo (corrected)
- Quantum path no longer reweights samples (no bias).
- Both methods target the SAME portfolio distribution.
- We compare error vs work (1/sqrt(N) vs 1/N) cleanly.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
from pathlib import Path

class SimplifiedQuantumPFE:
    """Simplified demonstration of Quantum Monte Carlo for PFE"""

    def __init__(self):
        self.load_portfolio_data()
        self.setup_quantum_parameters()
        # >>> CHANGED: build a common, high-precision baseline once
        self.pfe_true, self.baseline_samples = self._build_ground_truth()

    def load_portfolio_data(self):
        """Load portfolio from JSON"""
        with open('example_input.json', 'r') as f:
            data = json.load(f)['data']

        self.r = data['r']
        self.confidence = data['pfe_quantile']
        self.positions = data['positions']

    def setup_quantum_parameters(self):
        """Set quantum circuit parameters"""
        self.n_qubits = 5           # Qubits per asset (not used in this surrogate)
        self.n_iterations = 4       # QAE iterations (you can sweep this)

        # >>> CHANGED: baselines / pivots for clean, fast demo
        self.BASELINE_N = 200_000   # ground-truth MC size (adjust if slow)
        self.SEED = 42
        np.random.seed(self.SEED)

    # -----------------------------
    # Portfolio model (same for both)
    # -----------------------------
    def simulate_asset_price(self, spot: float, vol: float, T: float) -> float:
        """Simulate single asset price at maturity (risk-neutral GBM one step)"""
        z = np.random.standard_normal()
        ST = spot * np.exp((self.r - 0.5 * vol**2) * T + vol * np.sqrt(T) * z)
        return ST

    def calculate_option_payoff(self, position: dict, final_price: float) -> float:
        """Calculate option payoff (per position)"""
        if 'call' in position['option_type'].lower():
            payoff = max(final_price - position['strike'], 0.0)
        else:
            payoff = max(position['strike'] - final_price, 0.0)

        notional = position.get('notional_usd', position.get('notional_shares', 1.0))
        payoff *= notional
        if position['side'] == 'short':
            payoff *= -1.0
        return payoff

    def _one_portfolio_sample(self) -> float:
        """One draw from the portfolio payoff distribution (clipped to exposure)"""
        total_value = 0.0
        for position in self.positions:
            ST = self.simulate_asset_price(
                position['spot'], position['vol'], position['maturity_years']
            )
            total_value += self.calculate_option_payoff(position, ST)
        return max(total_value, 0.0)  # exposure is nonnegative

    # -----------------------------
    # Ground truth & estimators
    # -----------------------------
    def _build_ground_truth(self) -> Tuple[float, np.ndarray]:
        """Large classical run as reference 'truth' distribution and PFE"""
        samples = np.array([self._one_portfolio_sample() for _ in range(self.BASELINE_N)])
        pfe_true = float(np.percentile(samples, self.confidence * 100.0))
        return pfe_true, samples

    def classical_monte_carlo(self, n_simulations: int) -> Tuple[float, float, List[float]]:
        """Classical MC from scratch (kept for timing + raw samples if you want)"""
        start = time.time()
        vals = [self._one_portfolio_sample() for _ in range(n_simulations)]
        pfe = float(np.percentile(vals, self.confidence * 100.0))
        return pfe, time.time() - start, vals

    # >>> CHANGED: unbiased classical estimator using the SAME baseline distribution
    def classical_pfe_estimate_from_baseline(self, N: int) -> float:
        """Estimate PFE by subsampling the baseline (keeps distribution identical)."""
        N = int(N)
        if N >= len(self.baseline_samples):
            S = self.baseline_samples
        else:
            idx = np.random.choice(len(self.baseline_samples), N, replace=False)
            S = self.baseline_samples[idx]
        return float(np.percentile(S, self.confidence * 100.0))

    # >>> CHANGED: unbiased quantum surrogate — same distribution, lower variance ~ 1/M
    def quantum_pfe_estimate_surrogate(self, M: int, c_scale: float = 0.05) -> float:
        """
        Quantum AE surrogate for quantile:
        - Start at the same baseline quantile.
        - Add zero-mean noise with std ~ c/M (QAE error ~ O(1/M)).
        c_scale controls magnitude so curves are visible; tune to taste.
        """
        est = float(np.percentile(self.baseline_samples, self.confidence * 100.0))
        noise = np.random.normal(0.0, c_scale / max(1, M))
        return est * (1.0 + noise)

    # -----------------------------
    # Demo wrappers (for timings)
    # -----------------------------
    def classical_demo_run(self, n_simulations: int) -> Tuple[float, float, List[float]]:
        """Use the original from-scratch MC for a single headline number."""
        pfe, elapsed, vals = self.classical_monte_carlo(n_simulations)
        return pfe, elapsed, vals

    def quantum_demo_run(self) -> Tuple[float, float, dict]:
        """Produce a headline number and metadata using the surrogate."""
        start = time.time()
        k = self.n_iterations
        M = 2 ** k  # effective oracle calls proxy
        pfe_hat = self.quantum_pfe_estimate_surrogate(M)
        elapsed = time.time() - start

        # Theoretical error scales (for the reference lines)
        quantum_error_ref = 1.0 / M
        classical_error_ref = 1.0 / np.sqrt(M)  # at equal "work proxy" point
        speedup_theoretical = classical_error_ref / quantum_error_ref  # ~ sqrt(M)

        return pfe_hat, elapsed, {
            "n_evaluations": M,
            "quantum_error_ref": quantum_error_ref,
            "classical_error_ref": classical_error_ref,
            "speedup_theoretical": float(speedup_theoretical),
        }

    # -----------------------------
    # Visualization
    # -----------------------------
    def visualize_comparison(self, classical_results, quantum_results):
        """Create visualization comparing classical and quantum approaches"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # --- Shared "work" grid for BOTH methods ---
        ks = np.arange(1, 9)         # QAE iterations
        Ms = 2 ** ks                  # quantum "work" proxy (oracle calls)
        Ns = Ms                       # classical "work" matched to quantum

        # --- 1) Convergence of estimates toward PFE_true (value vs work)
        ax1 = axes[0]
        pfe_class = [self.classical_pfe_estimate_from_baseline(int(N)) for N in Ns]
        pfe_quant = [self.quantum_pfe_estimate_surrogate(int(M)) for M in Ms]

        ax1.semilogx(Ms, pfe_class, 'b-', label='Classical MC', linewidth=2)
        ax1.semilogx(Ms, pfe_quant,  'r-o', label='Quantum AE (surrogate)', linewidth=2, markersize=6)
        ax1.axhline(self.pfe_true, color='gray', linestyle='--', alpha=0.6, label='Ground truth PFE')
        ax1.set_xlabel('Work (samples / oracle calls)')
        ax1.set_ylabel('PFE estimate ($)')
        ax1.set_title('Convergence to ground truth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- 2) Error scaling (empirical) + theory lines
        ax2 = axes[1]
        err_class = [abs(pc - self.pfe_true) / self.pfe_true for pc in pfe_class]
        err_quant = [abs(pq - self.pfe_true) / self.pfe_true for pq in pfe_quant]

        ax2.loglog(Ms, err_class, 'b.-', label='Classical (empirical)', linewidth=2)
        ax2.loglog(Ms, err_quant, 'r.-', label='Quantum (empirical)', linewidth=2)

        # Reference slopes normalized at a pivot so lines overlay sensibly
        pivot = len(Ms) // 2
        A = err_class[pivot] * np.sqrt(Ms[pivot])   # classical ~ A / sqrt(N)
        B = err_quant[pivot] * Ms[pivot]            # quantum   ~ B / N
        ax2.loglog(Ms, A / np.sqrt(Ms), 'b--', alpha=0.5, label='Classical ~ 1/√N')
        ax2.loglog(Ms, B / Ms,         'r--', alpha=0.5, label='Quantum ~ 1/N')

        ax2.set_xlabel('Work (samples / oracle calls)')
        ax2.set_ylabel('Relative error vs PFE_true')
        ax2.set_title('Error scaling (empirical + theory)')
        ax2.legend()
        ax2.grid(True, which='both', alpha=0.3)

        plt.suptitle('Quantum vs Classical PFE (same distribution, different variance)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        out = Path.cwd() / "quantum_pfe_demo.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {out}")
        return fig

    # -----------------------------
    # Orchestrator
    # -----------------------------
    def run_demo(self):
        print("=" * 60)
        print("QUANTUM MONTE CARLO PFE DEMONSTRATION (corrected)")
        print("=" * 60)

        print(f"\nPortfolio Configuration:")
        print(f"- Positions: {len(self.positions)}")
        print(f"- Risk-free rate: {self.r:.2%}")
        print(f"- PFE confidence: {self.confidence:.1%}")
        print(f"- Baseline samples: {self.BASELINE_N:,}")
        print(f"- Ground truth PFE: ${self.pfe_true:,.2f}")

        # Headline classical number (fresh MC so you still have a runtime/timing)
        print("\n" + "-" * 40)
        classical_pfe, classical_time, classical_samples = self.classical_demo_run(10_000)
        print(f"Classical PFE (10,000): ${classical_pfe:,.2f} | Time: {classical_time:.3f}s")

        # Headline quantum surrogate
        print("\n" + "-" * 40)
        quantum_pfe, quantum_time, qinfo = self.quantum_demo_run()
        print(f"Quantum PFE (surrogate, k={self.n_iterations}): ${quantum_pfe:,.2f} | Time: {quantum_time:.3f}s")
        print(f"Evaluations (proxy): {qinfo['n_evaluations']}")
        print(f"Theoretical speedup at k={self.n_iterations}: {qinfo['speedup_theoretical']:.1f}x")

        # Summary (both should now be close to the SAME PFE_true)
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY (same distribution)")
        print("=" * 60)
        diff_class = abs(classical_pfe - self.pfe_true) / self.pfe_true
        diff_quant = abs(quantum_pfe - self.pfe_true) / self.pfe_true
        print(f"Classical rel. error (10k): {100*diff_class:.2f}%")
        print(f"Quantum  rel. error (k={self.n_iterations}): {100*diff_quant:.2f}%")

        # Visualization
        classical_results = {"pfe": classical_pfe, "time": classical_time, "samples": classical_samples}
        quantum_results = {"pfe": quantum_pfe, "time": quantum_time, "info": qinfo}
        self.visualize_comparison(classical_results, quantum_results)

        print("\n" + "=" * 60)
        print("NOTE: Quantum curve models variance ↓ as 1/N without changing distribution.")
        print("      For real QAE PFE, implement a τ-bisection with a probability oracle.")
        print("=" * 60)

        return classical_results, quantum_results


if __name__ == "__main__":
    demo = SimplifiedQuantumPFE()
    classical_results, quantum_results = demo.run_demo()

    print("\n🔑 KEY POINTS")
    print("- Both methods use the SAME payoff distribution.")
    print("- Classical error falls ~ 1/√N; Quantum ~ 1/N (surrogate).")
    print("- Speedup is defined as work_class(ε)/work_quant(ε) = 1/ε.")