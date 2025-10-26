import json

# --- paste your JSON dict here (unchanged) ---
raw = {
  "data": {
    "r": 0.02,
    "pfe_quantile": 0.95,
    "seed": 42,
    "ionq_shots": 200,
    "positions": [
      {"symbol":"USDJPY","asset_type":"fx","side":"long","option_type":"put","strike":141.0,"spot":140.0,"vol":0.15,"maturity_years":0.25,"notional_usd":5000000.0,"base_ccy":"USD","quote_ccy":"JPY"},
      {"symbol":"USDSGD","asset_type":"fx","side":"long","option_type":"put_usd_call_sgd","strike":1.29,"spot":1.30,"vol":0.12,"maturity_years":0.25,"notional_usd":2000000.0,"base_ccy":"USD","quote_ccy":"SGD"},
      {"symbol":"AAPL","asset_type":"equity","side":"long","option_type":"call","strike":210.0,"spot":250.0,"vol":0.22,"maturity_years":0.25,"notional_shares":10000},
      {"symbol":"TSLA","asset_type":"equity","side":"short","option_type":"put","strike":430.0,"spot":440.0,"vol":0.51,"maturity_years":0.25,"notional_shares":5000},
      {"symbol":"MSFT","asset_type":"equity","side":"long","option_type":"put","strike":530.0,"spot":520.0,"vol":0.14,"maturity_years":0.25,"notional_shares":2000},
      {"symbol":"AMZN","asset_type":"equity","side":"long","option_type":"call","strike":230.0,"spot":225.0,"vol":0.25,"maturity_years":0.25,"notional_shares":8000},
      {"symbol":"NVDA","asset_type":"equity","side":"short","option_type":"call","strike":190.0,"spot":180.0,"vol":0.32,"maturity_years":0.25,"notional_shares":1000}
    ]
  }
}

data = raw["data"]
r   = float(data["r"])
alpha_target = float(data["pfe_quantile"])
positions = data["positions"][3:6]  

print("Using first three positions:")
for p in positions:
    print(p["symbol"], p["asset_type"], p["side"], p["option_type"])

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Vectorized erf
try:
    from scipy.special import erf
except Exception:
    import math
    erf = np.vectorize(math.erf)

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
try:
    from qiskit_aer.primitives import Sampler  # fast local if present
except Exception:
    from qiskit.primitives import Sampler

# NORMAL_CDF:
# Returns the Normal **CDF** (cumulative distribution function):
# the probability that a Normal(mu, sigma^2) variable is ≤ x.
# Works for scalars or arrays.
def normal_cdf(x, mu, sigma):
    z = (np.asarray(x) - mu) / (sigma + 1e-16)  # tiny epsilon avoids divide-by-zero
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

# BIN_PROB_EDGES:
# Given bin edges [e0, e1, ..., eN], compute the probability in each bin
# under Normal(mu, sigma^2). We use differences of the CDF at the edges.
def bin_prob_edges(edges, mu, sigma):
    cdf_vals = normal_cdf(edges, mu, sigma)  # CDF at edges
    probs = np.diff(cdf_vals)                # bin prob = CDF(right) - CDF(left)
    return np.clip(probs, 0.0, None)         # clamp tiny negatives to 0

# MAKE_PRICE_GRID:
# Build an evenly spaced price grid and its bin probabilities.
# - S0: spot, r: rate, vol: volatility, tau: years
# - n_qubits -> number of bins is 2**n_qubits
# - Grid spans ±k * sigma_t around the mean mu
# - vol_is_relative=True means sigma_t = vol*S0*sqrt(tau); else vol*sqrt(tau)
# Returns: centers (bin centers), probs (per-bin probabilities, sum≈1), (Smin, Smax).
def make_price_grid(S0, r, vol, tau, n_qubits, k=5.0, vol_is_relative=True):
    mu = S0 * (1.0 + r * tau)
    sigma_t = (vol * S0 if vol_is_relative else vol) * np.sqrt(tau)

    n = 2 ** n_qubits
    Smin, Smax = mu - k * sigma_t, mu + k * sigma_t
    edges = np.linspace(Smin, Smax, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    probs = bin_prob_edges(edges, mu, sigma_t)
    probs = probs / (probs.sum() + 1e-16)  # normalize safely
    return centers, probs, (Smin, Smax)

# AMPS_FROM_PROBS:
# Convert nonnegative probabilities to a normalized amplitude vector
# (square roots of probs, then divide by the Euclidean norm).
def amps_from_probs(probs):
    a = np.sqrt(np.clip(np.asarray(probs, dtype=float), 0.0, None))
    return a / (np.linalg.norm(a) + 1e-16)

# PRICE_TO_INDEX:
# Map a real-world threshold S_th (e.g., K±y) to the **bin index** in the
# evenly spaced grid [Smin, Smax] with 2**n_qubits bins.
# Returns an integer in [0, 2**n_qubits - 1].
def price_to_index(S_th, Smin, Smax, n_qubits):
    n = 2 ** n_qubits
    edges = np.linspace(Smin, Smax, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.searchsorted(centers, S_th, side="right") - 1
    return max(0, min(n - 1, idx))


def price_to_index_le(S_th, Smin, Smax, n_qubits):
    """
    For a <= comparator: return the largest bin index whose RIGHT edge <= S_th.
    This avoids counting any partial bin above the threshold.
    """
    n = 2**n_qubits
    edges = np.linspace(Smin, Smax, n + 1)   # e0,...,eN
    # compare against RIGHT edges e1..eN
    idx = np.searchsorted(edges[1:], S_th, side="right") - 1
    return max(0, min(n - 1, idx))

def price_to_index_ge(S_th, Smin, Smax, n_qubits):
    """
    For a >= comparator: return the smallest bin index whose LEFT edge >= S_th.
    This avoids counting any partial bin below the threshold.
    """
    n = 2**n_qubits
    edges = np.linspace(Smin, Smax, n + 1)   # e0,...,eN
    # compare against LEFT edges e0..e(N-1)
    idx = np.searchsorted(edges[:-1], S_th, side="left")
    return max(0, min(n - 1, idx))

# _APPLY_MATCH_AND_MCX (helper):
# For a given basis integer x, temporarily flip price qubits so that |x>
# looks like |11...1>, apply a multi-controlled X on the flag, then undo flips.
def _apply_match_and_mcx(qc, n_qubits, x, flag_q):
    for b in range(n_qubits):
        if ((x >> b) & 1) == 0:
            qc.x(b)
    qc.mcx(list(range(n_qubits)), flag_q)
    for b in range(n_qubits):
        if ((x >> b) & 1) == 0:
            qc.x(b)


# ADD_COMPARATOR_LE:
# Mark all basis states |x> with x <= thr_idx by flipping the **flag** qubit.
# (Simple and clear, not depth-optimal; fine for small n_qubits.)
def add_comparator_le(qc, n_qubits, thr_idx, flag_q):
    for x in range(thr_idx + 1):
        _apply_match_and_mcx(qc, n_qubits, x, flag_q)


# ADD_COMPARATOR_GE:
# Mark all basis states |x> with x >= thr_idx by flipping the **flag** qubit.
def add_comparator_ge(qc, n_qubits, thr_idx, flag_q):
    for x in range(thr_idx, 2 ** n_qubits):
        _apply_match_and_mcx(qc, n_qubits, x, flag_q)


# BUILD_A_OPERATOR_EQUITY:
# Build the A-operator used by QAE:
#  1) load amplitudes for the price register (StatePreparation(amps))
#  2) flip the flag qubit on the "good" set {E <= y}
#     - CALL: E = max(S-K, 0) <= y  ⇔  S <= K + y   (use <= comparator)
#     - PUT : E = max(K-S, 0) <= y  ⇔  S >= K - y   (use >= comparator)
# Returns: (circuit, S_threshold, threshold_index)

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem

def build_A_operator_equity(n_qubits, amps, Smin, Smax, K, y, option_type="call"):
    qc = QuantumCircuit(n_qubits + 1, name="A")                 # +1 flag qubit
    qc.append(StatePreparation(np.asarray(amps, dtype=complex)), range(n_qubits))

    opt = option_type.lower()
    if opt == "call":
        S_th = K + y
        thr_idx = price_to_index_le(S_th, Smin, Smax, n_qubits)   # <= comparator uses RIGHT edges
        add_comparator_le(qc, n_qubits, thr_idx, n_qubits)
    elif "put" in opt:
        S_th = K - y
        thr_idx = price_to_index_ge(S_th, Smin, Smax, n_qubits)   # >= comparator uses LEFT edges
        add_comparator_ge(qc, n_qubits, thr_idx, n_qubits)
    else:
        raise ValueError("unknown option_type (use 'call' or something containing 'put')")

    return qc, S_th, thr_idx


# QAE_PROBABILITY:
# Run Iterative Amplitude Estimation (IAE) to estimate Prob(flag = 1) for circuit qc.
def qae_probability(qc, sampler, epsilon_target=0.02, alpha=0.05):
    problem = EstimationProblem(
        state_preparation=qc,
        objective_qubits=[qc.num_qubits - 1],   # last qubit is the flag
        grover_operator=None,                   # let IAE build Grover internally
    )
    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon_target, alpha=alpha, sampler=sampler
    )
    return float(iae.estimate(problem).estimation)


# FIND_PFE_QUANTILE_QAE:
# Find y such that P(E <= y) ≈ alpha_target using **bisection**.
# At each mid-y, build A with the right comparator and call QAE to get P(flag=1).
def find_pfe_quantile_qae(
    alpha_target, K, n_qubits, amps, Smin, Smax,
    y_low, y_high, option_type="call",
    max_iter=12, iae_eps=0.02, iae_alpha=0.05,
    sampler=None
):
    if sampler is None:
        raise ValueError("Please pass a sampler (e.g., algo_sampler) to find_pfe_quantile_qae().")

    low, high = y_low, y_high
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        qc_mid, _, _ = build_A_operator_equity(
            n_qubits, amps, Smin, Smax, K, y=mid, option_type=option_type
        )
        p = qae_probability(qc_mid, sampler, epsilon_target=iae_eps, alpha=iae_alpha)
        if p < alpha_target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)

# === SAFE sampler for shot-based AE (prevents kernel crashes) ===
# Use this instead of RefSampler/AerSampler directly.

import os
# ↓ Do this *before* importing qiskit_aer to avoid thread fights that can crash kernels
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def make_sampler_safe(shots: int = 256, seed: int = 7):
    """
    Returns (sampler, note) where sampler provides a Sampler-compatible API
    for qiskit_algorithms.AmplitudeEstimation, and 'note' explains which path we used.
    Preference order:
      1) BackendSampler(AerSimulator)  [most stable]
      2) Aer legacy Sampler            [still stable]
      3) RefSampler + binomial noise   [fallback that mimics shots]
    """
    # 1) BackendSampler on AerSimulator (most stable)
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import BackendSampler

        backend = AerSimulator(method="automatic")
        # hard-limit threading to avoid crashes on some stacks
        try:
            backend.set_options(max_parallel_threads=1, max_parallel_experiments=1)
        except Exception:
            pass

        sampler = BackendSampler(
            backend=backend,
            options={"shots": int(shots), "seed_simulator": int(seed)}
        )
        return sampler, "BackendSampler(AerSimulator)"
    except Exception as e_bs:
        last_err = f"BackendSampler unavailable: {e_bs}"

    # 2) Aer legacy Sampler
    try:
        from qiskit_aer.primitives import Sampler as AerSamplerLegacy
        sampler = AerSamplerLegacy()
        try:
            sampler.set_options(shots=int(shots), seed_simulator=int(seed))
        except Exception:
            sampler.options.shots = int(shots)
            sampler.options.seed_simulator = int(seed)
        return sampler, "Aer legacy Sampler"
    except Exception as e_legacy:
        last_err = f"{last_err} | Aer legacy Sampler unavailable: {e_legacy}"

    # 3) Fallback: analytic RefSampler + artificial shot noise (binomial resample)
    #    AE needs a Sampler-like object. We wrap RefSampler and inject binomial noise
    #    into the outcome probabilities to emulate 'shots'.
    from qiskit.primitives import Sampler as RefSampler
    import numpy as np

    class NoisyRefSampler:
        def __init__(self, shots, seed):
            self.ref = RefSampler()
            self.shots = int(shots)
            self.rng = np.random.default_rng(int(seed))
            # expose options attr to satisfy some algorithms
            class _Opt: pass
            self.options = _Opt()
            self.options.shots = int(shots)

        def run(self, circuits, parameter_values=None, **kwargs):
            job = self.ref.run(circuits, parameter_values=parameter_values, **kwargs)
            res = job.result()

            # Inject multinomial sampling noise into quasi-probs:
            from qiskit.result import SamplerResult
            quasi = []
            metadata = []
            for q in res.quasi_dists:
                # q is a dict bitstring->p; draw counts ~ Multinomial(shots, p)
                keys = list(q.keys())
                probs = np.array([q[k] for k in keys], float)
                probs = np.clip(probs, 0, None)
                probs /= probs.sum() + 1e-16
                counts = self.rng.multinomial(self.shots, probs)
                noisy = {k: c / max(1, self.shots) for k, c in zip(keys, counts)}
                quasi.append(noisy)
                metadata.append({"shots": self.shots})
            # mimic SamplerResult
            return SamplerResult(quasi_dists=quasi, metadata=metadata)

    sampler = NoisyRefSampler(shots=int(shots), seed=int(seed))
    return sampler, f"RefSampler+binomial noise (fallback). Warning: Aer unavailable. {last_err}"

# KNOBS (model/discretization/QAE settings)
n_price       = 4        # number of price qubits -> 2**n_price bins
k_sigma       = 5.0      # grid spans ± k * sigma_t around the mean
iae_eps       = 0.02     # QAE target additive error (smaller -> more work)
iae_alpha     = 0.05     # QAE confidence level (≈ 95%)
vol_is_relative = True   # vol is fraction of spot (e.g., 0.22)

# <<< ADDED: stable reference sampler reused across all QAE calls
from qiskit.primitives import Sampler as RefSampler
algo_sampler = RefSampler()
algo_sampler.set_options(shots=256, seed_simulator=7)

# OUTPUT CONTAINERS
results  = []   # rows for the summary table
per_plot = []   # per-instrument data for CDF plots

# MAIN LOOP OVER THE FIRST-THREE POSITIONS
for p in positions:
    # BASIC FIELDS (symbol, types)
    sym   = p["symbol"]
    atype = p["asset_type"].lower()
    otype = p["option_type"].lower()
    type_norm = "put" if "put" in otype else "call"   # keep your threshold logic

    # INPUTS FOR THIS INSTRUMENT (spot, strike, vol, maturity)
    S0  = float(p["spot"])
    K   = float(p["strike"])
    vol = float(p["vol"])
    tau = float(p["maturity_years"])

    # DISCRETIZE TERMINAL PRICE AND GET AMPLITUDES
    # S  : price bin centers
    # q  : per-bin probabilities (sum to 1)
    # Smin/Smax : grid bounds
    S, q, (Smin, Smax) = make_price_grid(
        S0, r, vol, tau, n_price, k=k_sigma, vol_is_relative=vol_is_relative
    )
    amps = np.asarray(amps_from_probs(q), dtype=complex)  # for StatePreparation

    # MAP PRICE -> EXPOSURE (per-unit) AND SET BISECTION BOUNDS FOR y
    if type_norm == "call":
        exposure = np.maximum(S - K, 0.0)
        y_low, y_high = 0.0, max(0.0, Smax - K)
        units = "per-share" if atype == "equity" else "rate-units"
    else:
        exposure = np.maximum(K - S, 0.0)
        y_low, y_high = 0.0, max(0.0, K - Smin)
        units = "per-share" if atype == "equity" else "rate-units"

    # QUANTUM AMPLITUDE ESTIMATION INSIDE BISECTION TO GET PFE AT α
    y_hat = find_pfe_quantile_qae(
        alpha_target, K, n_price, amps, Smin, Smax,
        y_low=y_low, y_high=y_high,
        option_type=type_norm,
        max_iter=12,            
        iae_eps=iae_eps, iae_alpha=iae_alpha,
        sampler=algo_sampler     # <<< FIXED typo: algo_sampler
    )

    # BUILD A CDF (for plotting) FROM DISCRETE EXPOSURE & PROBS
    order = np.argsort(exposure)
    exp_sorted = exposure[order]
    cdf_sorted = np.cumsum(q[order])

    # SCALE TO TRADE-LEVEL PFE USING NOTIONAL (SIMPLE LINEAR SCALING)
    notional = p.get("notional_shares", p.get("notional_usd"))
    if atype == "equity":
        pfe_trade, trade_units = y_hat * float(notional), "USD"
    else:
        pfe_trade, trade_units = y_hat * float(notional), "USD (approx)"

    # RECORD ROW FOR THE SUMMARY TABLE
    results.append({
        "symbol": sym,
        "asset_type": atype,
        "option_type": otype,
        "side": p["side"],
        "S0": S0, "K": K, "vol": vol, "tau_years": tau,
        "n_price_qubits": n_price,
        "PFE95_per_unit": y_hat,
        "per_unit_units": units,
        "notional": notional,
        "PFE95_trade_level": pfe_trade,
        "trade_units": trade_units,
    })

    # SAVE SERIES FOR CDF PLOT
    per_plot.append((sym, type_norm, exp_sorted, cdf_sorted, y_hat))

# STATUS MESSAGE
print("Finished QAE runs for:", [row["symbol"] for row in results])

df = pd.DataFrame(results, columns=[
    "symbol","asset_type","option_type","side","S0","K","vol","tau_years",
    "n_price_qubits","PFE95_per_unit","per_unit_units","notional","PFE95_trade_level","trade_units"
])

print(df.to_string(index=False))

csv_path = "pfe_qae_first3_summary.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved summary to {csv_path}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (sym, type_norm, x, c, yhat) in zip(axes, per_plot):
    ax.plot(x, c, drawstyle='steps-post')
    ax.axhline(alpha_target, color='orange', linestyle='--', alpha=0.8, label=f"α={alpha_target}")
    ax.axvline(yhat, color='red', linestyle='--', alpha=0.9, label='PFE95 (QAE)')
    ax.set_title(f"{sym} ({type_norm.upper()})")
    ax.set_xlabel("Exposure (per-unit)")
    ax.set_ylabel("P(E ≤ y)")
    ax.grid(True)
    ax.legend(loc="lower right")
plt.suptitle("Exposure CDFs with QAE PFE95 markers", y=1.05)
plt.tight_layout()
plt.show()

# ===== Log–log EE convergence plots for three positions (multi-trial median + IQR) =====
import numpy as np
import matplotlib.pyplot as plt

data = raw["data"]
r = float(data["r"])
positions = data["positions"][3:6]   # choose which 3 you want (e.g., TSLA, MSFT, AMZN)

# --- knobs (same as your pipeline) ---
n_price = 5          # price qubits -> 2**n_price bins
k_sigma = 5.0
vol_is_relative = True

# sampling schedule
N_max = 50_000
steps = np.unique(np.concatenate([
    np.arange(100, 2_000, 100),
    np.arange(2_000, 10_000, 500),
    np.arange(10_000, N_max+1, 2_000)
]))

# multi-trial settings
B = 40                      # number of independent trials per instrument
base_seed = 2025            # reproducible; trials use base_seed + b

# If helpers not already in scope, define minimal versions
try:
    make_price_grid
    amps_from_probs
except NameError:
    try:
        from scipy.special import erf
    except Exception:
        import math
        erf = np.vectorize(math.erf)

    def normal_cdf(x, mu, sigma):
        z = (np.asarray(x) - mu) / (sigma + 1e-16)
        return 0.5 * (1 + erf(z / np.sqrt(2)))

    def bin_prob_edges(edges, mu, sigma):
        cdfs = normal_cdf(edges, mu, sigma)
        return np.maximum(0.0, np.diff(cdfs))

    def make_price_grid(S0, r, vol, tau, n_qubits, k=5.0, vol_is_relative=True):
        mu = S0 * (1 + r * tau)
        sigma_t = (vol * S0) * np.sqrt(tau) if vol_is_relative else vol * np.sqrt(tau)
        n = 2**n_qubits
        Smin, Smax = mu - k*sigma_t, mu + k*sigma_t
        edges = np.linspace(Smin, Smax, n+1)
        centers = 0.5*(edges[:-1] + edges[1:])
        probs = bin_prob_edges(edges, mu, sigma_t)
        probs = probs / probs.sum()
        return centers, probs, (Smin, Smax)

for p in positions:
    sym   = p["symbol"]
    atype = p["asset_type"].lower()
    otype = p["option_type"].lower()
    type_norm = "put" if "put" in otype else "call"

    S0  = float(p["spot"])
    K   = float(p["strike"])
    vol = float(p["vol"])
    tau = float(p["maturity_years"])

    # Discretize terminal price (your model), then map to exposure per unit
    S, q, (Smin, Smax) = make_price_grid(S0, r, vol, tau, n_price, k=k_sigma, vol_is_relative=vol_is_relative)
    exposure_vals = np.maximum(S - K, 0.0) if type_norm == "call" else np.maximum(K - S, 0.0)
    probs = q

    # "True" EE on the discretized PMF (baseline for convergence)
    EE_true = float(np.sum(exposure_vals * probs))
    EE_true = max(EE_true, 1e-16)  # avoid log(0)

    # Storage for trials
    means_trials   = np.empty((B, len(steps)), dtype=float)
    relerr_trials  = np.empty((B, len(steps)), dtype=float)

    for b in range(B):
        rng = np.random.default_rng(base_seed + b)
        samples = rng.choice(exposure_vals, size=N_max, p=probs, replace=True)

        running_sum  = np.cumsum(samples, dtype=float)
        running_mean = running_sum[steps-1] / steps
        running_mean = np.maximum(running_mean, 1e-16)

        abs_error = np.abs(running_mean - EE_true)
        rel_error = np.maximum(abs_error / EE_true, 1e-16)

        means_trials[b, :]  = running_mean
        relerr_trials[b, :] = rel_error

    # Aggregate across trials: median + IQR (25–75%)
    mean_med = np.median(means_trials, axis=0)
    mean_q25 = np.percentile(means_trials, 25, axis=0)
    mean_q75 = np.percentile(means_trials, 75, axis=0)

    rel_med = np.median(relerr_trials, axis=0)
    rel_q25 = np.percentile(relerr_trials, 25, axis=0)
    rel_q75 = np.percentile(relerr_trials, 75, axis=0)

    # 1/sqrt(n) guide (anchored at median's first point)
    ref_c = rel_med[0] * np.sqrt(steps[0])
    ref_line = ref_c / np.sqrt(steps)

    # ---- Plots (log–log) per instrument ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(f"{sym}  ({type_norm.upper()}) — EE convergence (median over {B} trials)", y=1.04)

    # (1) EE vs samples (median + IQR)
    ax = axes[0]
    ax.loglog(steps, mean_med, lw=2.0, label="Median sample mean EE")
    ax.fill_between(steps, mean_q25, mean_q75, alpha=0.20, label="25–75% across trials")
    ax.loglog([steps[0], steps[-1]], [EE_true, EE_true], 'g--', label=f"True EE = {EE_true:.4f}")
    ax.set_xlabel("Number of samples (log)")
    ax.set_ylabel("Expected Exposure (log)")
    ax.set_title("EE vs samples (log–log)")
    ax.grid(True, which='both')
    ax.legend()

    # (2) Relative error vs samples (median + IQR) + guide
    ax = axes[1]
    ax.loglog(steps, rel_med, lw=2.0, label="Median relative error")
    ax.fill_between(steps, rel_q25, rel_q75, alpha=0.20, label="25–75% across trials")
    ax.loglog(steps, ref_line, '--', label="~ 1/n guide")
    ax.set_xlabel("Number of samples (log)")
    ax.set_ylabel("Relative error (log)")
    ax.set_title("EE relative error (log–log)")
    ax.grid(True, which='both')
    ax.legend()

    plt.tight_layout()
    plt.show()

# === Diagnostics: Portfolio Exposure Histogram + MC Convergence (drop at end) ===
import numpy as np, matplotlib.pyplot as plt

# --- knobs for the plots ---
try:
    alpha_plot = float(alpha_target)
except NameError:
    alpha_plot = 0.95

N_ref         = 300_000      # reference Monte Carlo sample size for "truth"
bins_hist     = 120          # histogram bins
seed          = 42           # reproducibility
k_sigma_plot  = float(k_sigma) if "k_sigma" in globals() else 6.0

rng = np.random.default_rng(seed)

# --- build discrete price models per instrument (using your current n_price grid) ---
models = []  # list of dicts per position with (S, q, K, otype, notional)
for p in positions:
    S0  = float(p["spot"])
    K   = float(p["strike"])
    vol = float(p["vol"])
    tau = float(p["maturity_years"])
    otype = "put" if "put" in p["option_type"].lower() else "call"
    atype = p["asset_type"].lower()
    notional = p.get("notional_shares", p.get("notional_usd"))

    S, q, (Smin, Smax) = make_price_grid(
        S0, r, vol, tau, n_price, k=k_sigma_plot, vol_is_relative=vol_is_relative
    )

    models.append({
        "S": S, "q": q, "K": K, "otype": otype,
        "scale": float(notional) if notional is not None else 1.0
    })

# --- draw a large joint sample of portfolio exposure ---
# We sample each instrument independently from its discrete distribution, then sum exposures.
exposures = np.zeros(N_ref, dtype=float)

for m in models:
    S, q, K, otype, scale = m["S"], m["q"], m["K"], m["otype"], m["scale"]
    idx = rng.choice(len(S), size=N_ref, p=q)
    S_samp = S[idx]
    if otype == "call":
        exp_i = np.maximum(S_samp - K, 0.0)
    else:
        exp_i = np.maximum(K - S_samp, 0.0)
    exposures += scale * exp_i

# --- EE and PFE from the empirical distribution ---
EE_ref   = float(exposures.mean())
PFE_ref  = float(np.quantile(exposures, alpha_plot))

# --- Plot 1: Portfolio Exposure Distribution + markers ---
plt.figure(figsize=(6.6, 4.2), dpi=110)
plt.hist(exposures, bins=bins_hist, density=True, alpha=0.6)
plt.axvline(EE_ref,  color="tab:red",   linewidth=2.5, label="EE_T (mean)")
plt.axvline(PFE_ref, color="orange",    linewidth=2.5, label=f"PFE_{int(100*alpha_plot)}%")
plt.title("Portfolio Exposure Distribution at T")
plt.xlabel("Exposure (USD)")
plt.ylabel("Probability Density")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Convergence of Expected Exposure (MC error vs samples) ---
# use one permutation so partial prefixes emulate growing sample sizes
perm = rng.permutation(N_ref)
exp_perm = exposures[perm]

# choose checkpoints (roughly 20 points up to N_ref)
checkpoints = np.unique(np.linspace(1000, N_ref, 20, dtype=int))
rel_err = []

running_sum = 0.0
k = 0
errs = []
for n in checkpoints:
    # accumulate incrementally for speed
    while k < n:
        running_sum += exp_perm[k]
        k += 1
    EE_n = running_sum / n
    rel_err.append(abs(EE_n - EE_ref) / (abs(EE_ref) + 1e-16))

plt.figure(figsize=(6.6, 4.0), dpi=110)
plt.plot(checkpoints, rel_err)
plt.xlabel("Number of samples")
plt.ylabel("Relative error")
plt.title("Monte Carlo Convergence of Expected Exposure")
plt.tight_layout()
plt.show()

print(f"EE_T (mean) ≈ {EE_ref:,.2f} USD    |   PFE_{int(100*alpha_plot)} ≈ {PFE_ref:,.2f} USD")

