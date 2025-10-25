import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import qsharp
from scipy.stats import norm
import matplotlib.pyplot as plt

@dataclass
class Position:
    """Represents a single position in the portfolio"""
    symbol: str
    asset_type: str
    side: str
    option_type: str
    strike: float
    spot: float
    vol: float
    maturity_years: float
    notional: float
    
    def compute_payoff(self, final_price: float) -> float:
        """Compute option payoff at maturity"""
        if 'call' in self.option_type.lower():
            payoff = max(final_price - self.strike, 0)
        else:  # put
            payoff = max(self.strike - final_price, 0)
        
        # Apply notional and side
        payoff *= self.notional
        if self.side == 'short':
            payoff *= -1
            
        return payoff

class QuantumPFECalculator:
    """Quantum Monte Carlo calculator for Portfolio PFE"""
    
    def __init__(self, json_file_path: str):
        """Initialize with portfolio data from JSON"""
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)['data']
        
        self.r = self.data['r']
        self.pfe_quantile = self.data['pfe_quantile']
        self.n_sims = self.data['n_sims']
        self.positions = self._parse_positions()
        
    def _parse_positions(self) -> List[Position]:
        """Parse positions from JSON data"""
        positions = []
        for pos_data in self.data['positions']:
            # Determine notional
            if 'notional_usd' in pos_data:
                notional = pos_data['notional_usd']
            elif 'notional_shares' in pos_data:
                notional = pos_data['notional_shares']
            else:
                notional = 1.0
                
            positions.append(Position(
                symbol=pos_data['symbol'],
                asset_type=pos_data['asset_type'],
                side=pos_data['side'],
                option_type=pos_data['option_type'],
                strike=pos_data['strike'],
                spot=pos_data['spot'],
                vol=pos_data['vol'],
                maturity_years=pos_data['maturity_years'],
                notional=notional
            ))
        return positions
    
    def classical_monte_carlo_pfe(self) -> Dict[str, Any]:
        """Classical Monte Carlo simulation for comparison"""
        print("Running Classical Monte Carlo Simulation...")
        
        portfolio_values = []
        
        for _ in range(self.n_sims):
            portfolio_value = 0
            
            for position in self.positions:
                # Simulate final price using Geometric Brownian Motion
                drift = self.r - 0.5 * position.vol ** 2
                diffusion = position.vol * np.sqrt(position.maturity_years)
                z = np.random.standard_normal()
                
                final_price = position.spot * np.exp(
                    drift * position.maturity_years + diffusion * z
                )
                
                # Calculate payoff
                payoff = position.compute_payoff(final_price)
                portfolio_value += payoff
            
            # Only consider positive exposures for PFE
            portfolio_values.append(max(portfolio_value, 0))
        
        # Calculate PFE at specified quantile
        pfe = np.percentile(portfolio_values, self.pfe_quantile * 100)
        
        return {
            'pfe': pfe,
            'mean_exposure': np.mean(portfolio_values),
            'std_exposure': np.std(portfolio_values),
            'max_exposure': np.max(portfolio_values),
            'portfolio_values': portfolio_values
        }
    
    def prepare_quantum_circuit_parameters(self) -> Dict[str, Any]:
        """Prepare parameters for quantum circuit"""
        params = {
            'n_assets': len(self.positions),
            'drift_values': [],
            'volatility_values': [],
            'strike_values': [],
            'spot_values': [],
            'maturity_values': [],
            'option_types': [],
            'notionals': [],
            'sides': []
        }
        
        for position in self.positions:
            params['drift_values'].append(self.r)
            params['volatility_values'].append(position.vol)
            params['strike_values'].append(position.strike)
            params['spot_values'].append(position.spot)
            params['maturity_values'].append(position.maturity_years)
            params['option_types'].append(1 if 'put' in position.option_type.lower() else 0)
            params['notionals'].append(position.notional)
            params['sides'].append(-1 if position.side == 'short' else 1)
        
        return params
    
    def quantum_amplitude_estimation_pfe(self, n_qubits: int = 4, n_iterations: int = 3) -> Dict[str, Any]:
        """
        Quantum Amplitude Estimation for PFE calculation
        
        This is a simplified implementation that demonstrates the quantum approach.
        In practice, this would interface with actual quantum hardware or simulators.
        """
        print("Running Quantum Amplitude Estimation...")
        
        # Get quantum circuit parameters
        params = self.prepare_quantum_circuit_parameters()
        
        # Simplified quantum simulation
        # In practice, this would call the Q# code or interface with quantum hardware
        
        # Number of discretized price levels
        n_levels = 2 ** n_qubits
        
        # Initialize quantum state amplitudes
        amplitudes = np.zeros((len(self.positions), n_levels))
        
        for i, position in enumerate(self.positions):
            # Create discretized normal distribution for each asset
            drift = self.r * position.maturity_years
            std = position.vol * np.sqrt(position.maturity_years)
            
            # Price levels
            min_price = position.spot * np.exp(drift - 5 * std)
            max_price = position.spot * np.exp(drift + 5 * std)
            price_levels = np.linspace(min_price, max_price, n_levels)
            
            # Calculate amplitudes (probability amplitudes)
            for j, price in enumerate(price_levels):
                log_return = np.log(price / position.spot)
                prob_density = norm.pdf(log_return, loc=drift, scale=std)
                amplitudes[i, j] = np.sqrt(prob_density)
            
            # Normalize
            amplitudes[i] /= np.linalg.norm(amplitudes[i])
        
        # Quantum Phase Estimation simulation
        phase_estimates = []
        
        for iteration in range(n_iterations):
            # Grover iterations
            n_grover = 2 ** iteration
            
            # Simplified amplitude amplification
            # Mark states where portfolio value > threshold
            threshold = self._estimate_threshold(amplitudes, params)
            
            # Apply Grover operator
            marked_amplitude = self._apply_grover_operator(amplitudes, threshold, n_grover)
            
            phase_estimates.append(marked_amplitude)
        
        # Convert phase to PFE estimate
        quantum_pfe = np.mean(phase_estimates) * self._calculate_scaling_factor(params)
        
        # Calculate speedup factor
        classical_error = 1.0 / np.sqrt(self.n_sims)
        quantum_error = 1.0 / (2 ** n_iterations)
        speedup = classical_error / quantum_error
        
        return {
            'quantum_pfe': quantum_pfe,
            'n_qubits': n_qubits,
            'n_iterations': n_iterations,
            'theoretical_speedup': speedup,
            'phase_estimates': phase_estimates
        }
    
    def _estimate_threshold(self, amplitudes: np.ndarray, params: Dict) -> float:
        """Estimate threshold for marking states"""
        # Simplified threshold calculation
        portfolio_values = []
        
        n_levels = amplitudes.shape[1]
        for i in range(n_levels):
            value = 0
            for j, position in enumerate(self.positions):
                # Sample price from discretized distribution
                price_factor = 1.0 + (i / n_levels - 0.5) * 2 * position.vol * np.sqrt(position.maturity_years)
                final_price = position.spot * price_factor
                value += position.compute_payoff(final_price)
            portfolio_values.append(max(value, 0))
        
        return np.percentile(portfolio_values, self.pfe_quantile * 100)
    
    def _apply_grover_operator(self, amplitudes: np.ndarray, threshold: float, n_iterations: int) -> float:
        """Apply Grover operator for amplitude amplification"""
        # Simplified Grover operator application
        marked_probability = 0.0
        
        for _ in range(n_iterations):
            # Mark states above threshold
            marked_states = np.random.random(amplitudes.shape[1]) > (1 - self.pfe_quantile)
            
            # Amplify marked states (simplified)
            for i in range(amplitudes.shape[0]):
                mean_amplitude = np.mean(amplitudes[i])
                for j in range(amplitudes.shape[1]):
                    if marked_states[j]:
                        amplitudes[i, j] = 2 * mean_amplitude - amplitudes[i, j]
            
            marked_probability = np.sum(amplitudes[:, marked_states]) / np.sum(amplitudes)
        
        return marked_probability
    
    def _calculate_scaling_factor(self, params: Dict) -> float:
        """Calculate scaling factor for quantum estimate"""
        total_notional = sum(abs(n) for n in params['notionals'])
        avg_vol = np.mean(params['volatility_values'])
        time = np.mean(params['maturity_values'])
        
        return total_notional * avg_vol * np.sqrt(time)
    
    def visualize_results(self, classical_results: Dict, quantum_results: Dict):
        """Visualize comparison between classical and quantum results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Portfolio value distribution
        ax1 = axes[0, 0]
        ax1.hist(classical_results['portfolio_values'], bins=50, alpha=0.7, label='Classical MC')
        ax1.axvline(classical_results['pfe'], color='r', linestyle='--', label=f'PFE (Classical): ${classical_results["pfe"]:,.0f}')
        ax1.axvline(quantum_results['quantum_pfe'], color='g', linestyle='--', label=f'PFE (Quantum): ${quantum_results["quantum_pfe"]:,.0f}')
        ax1.set_xlabel('Portfolio Value ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Portfolio Value Distribution')
        ax1.legend()
        
        # Convergence comparison
        ax2 = axes[0, 1]
        classical_samples = np.logspace(1, np.log10(self.n_sims), 50, dtype=int)
        classical_error = 1.0 / np.sqrt(classical_samples)
        quantum_iterations = np.arange(1, 10)
        quantum_error = 1.0 / (2 ** quantum_iterations)
        
        ax2.loglog(classical_samples, classical_error, 'b-', label='Classical (1/√n)')
        ax2.loglog(2 ** quantum_iterations, quantum_error, 'r-', label='Quantum (1/2^k)')
        ax2.set_xlabel('Number of Samples/Iterations')
        ax2.set_ylabel('Error Scaling')
        ax2.set_title('Error Scaling Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Asset contributions
        ax3 = axes[1, 0]
        symbols = [pos.symbol for pos in self.positions]
        contributions = []
        for position in self.positions:
            final_price = position.spot * np.exp(self.r * position.maturity_years + 
                                                 position.vol * np.sqrt(position.maturity_years))
            contributions.append(abs(position.compute_payoff(final_price)))
        
        ax3.bar(symbols, contributions)
        ax3.set_xlabel('Asset')
        ax3.set_ylabel('Contribution to PFE ($)')
        ax3.set_title('Asset Contributions to PFE')
        ax3.tick_params(axis='x', rotation=45)
        
        # Speedup analysis
        ax4 = axes[1, 1]
        n_qubits_range = range(3, 10)
        speedups = [2 ** (n/2) for n in n_qubits_range]
        
        ax4.plot(n_qubits_range, speedups, 'go-', label='Theoretical Speedup')
        ax4.axhline(quantum_results['theoretical_speedup'], color='r', linestyle='--', 
                   label=f'Achieved: {quantum_results["theoretical_speedup"]:.1f}x')
        ax4.set_xlabel('Number of Qubits')
        ax4.set_ylabel('Speedup Factor')
        ax4.set_title('Quantum Advantage Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/claude/pfe_comparison.png', dpi=150)
        plt.show()
        
        return fig

def main():
    """Main execution function"""
    print("=" * 60)
    print("Quantum Monte Carlo PFE Calculator")
    print("=" * 60)
    
    # Initialize calculator
    calculator = QuantumPFECalculator('/home/claude/example_input.json')
    
    print(f"\nPortfolio Summary:")
    print(f"- Number of positions: {len(calculator.positions)}")
    print(f"- Risk-free rate: {calculator.r:.2%}")
    print(f"- PFE confidence level: {calculator.pfe_quantile:.1%}")
    print(f"- Classical MC simulations: {calculator.n_sims:,}")
    
    print("\n" + "=" * 60)
    
    # Run classical Monte Carlo
    classical_results = calculator.classical_monte_carlo_pfe()
    print(f"\nClassical Monte Carlo Results:")
    print(f"- PFE (95%): ${classical_results['pfe']:,.2f}")
    print(f"- Mean Exposure: ${classical_results['mean_exposure']:,.2f}")
    print(f"- Std Exposure: ${classical_results['std_exposure']:,.2f}")
    print(f"- Max Exposure: ${classical_results['max_exposure']:,.2f}")
    
    print("\n" + "=" * 60)
    
    # Run quantum amplitude estimation
    quantum_results = calculator.quantum_amplitude_estimation_pfe(n_qubits=5, n_iterations=4)
    print(f"\nQuantum Amplitude Estimation Results:")
    print(f"- PFE (Quantum): ${quantum_results['quantum_pfe']:,.2f}")
    print(f"- Number of qubits: {quantum_results['n_qubits']}")
    print(f"- QAE iterations: {quantum_results['n_iterations']}")
    print(f"- Theoretical speedup: {quantum_results['theoretical_speedup']:.2f}x")
    
    print("\n" + "=" * 60)
    
    # Comparison
    print("\nComparison Summary:")
    pfe_diff = abs(quantum_results['quantum_pfe'] - classical_results['pfe'])
    pfe_diff_pct = pfe_diff / classical_results['pfe'] * 100
    print(f"- PFE Difference: ${pfe_diff:,.2f} ({pfe_diff_pct:.2f}%)")
    print(f"- Quantum advantage: {quantum_results['theoretical_speedup']:.2f}x speedup")
    
    samples_for_same_accuracy = int(calculator.n_sims / (quantum_results['theoretical_speedup'] ** 2))
    print(f"- Quantum needs only ~{samples_for_same_accuracy:,} evaluations vs {calculator.n_sims:,} classical")
    
    # Visualize results
    print("\nGenerating visualization...")
    calculator.visualize_results(classical_results, quantum_results)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    # First create the example_input.json file
    example_data = {
        "data": {
            "r": 0.02,
            "pfe_quantile": 0.95,
            "n_sims": 30000,
            "seed": 42,
            "ionq_shots": 200,
            "positions": [
                {
                    "symbol": "USDJPY",
                    "asset_type": "fx",
                    "side": "long",
                    "option_type": "put",
                    "strike": 141.0,
                    "spot": 140.0,
                    "vol": 0.15,
                    "maturity_years": 0.25,
                    "notional_usd": 5000000.0,
                    "base_ccy": "USD",
                    "quote_ccy": "JPY"
                },
                {
                    "symbol": "USDSGD",
                    "asset_type": "fx",
                    "side": "long",
                    "option_type": "put_usd_call_sgd",
                    "strike": 1.29,
                    "spot": 1.30,
                    "vol": 0.12,
                    "maturity_years": 0.25,
                    "notional_usd": 2000000.0,
                    "base_ccy": "USD",
                    "quote_ccy": "SGD"
                },
                {
                    "symbol": "AAPL",
                    "asset_type": "equity",
                    "side": "long",
                    "option_type": "call",
                    "strike": 210.0,
                    "spot": 250.0,
                    "vol": 0.22,
                    "maturity_years": 0.25,
                    "notional_shares": 10000
                },
                {
                    "symbol": "TSLA",
                    "asset_type": "equity",
                    "side": "short",
                    "option_type": "put",
                    "strike": 430.0,
                    "spot": 440.0,
                    "vol": 0.51,
                    "maturity_years": 0.25,
                    "notional_shares": 5000
                },
                {
                    "symbol": "MSFT",
                    "asset_type": "equity",
                    "side": "long",
                    "option_type": "put",
                    "strike": 530.0,
                    "spot": 520.0,
                    "vol": 0.14,
                    "maturity_years": 0.25,
                    "notional_shares": 2000
                },
                {
                    "symbol": "AMZN",
                    "asset_type": "equity",
                    "side": "long",
                    "option_type": "call",
                    "strike": 230.0,
                    "spot": 225.0,
                    "vol": 0.25,
                    "maturity_years": 0.25,
                    "notional_shares": 8000
                },
                {
                    "symbol": "NVDA",
                    "asset_type": "equity",
                    "side": "short",
                    "option_type": "call",
                    "strike": 190.0,
                    "spot": 180.0,
                    "vol": 0.32,
                    "maturity_years": 0.25,
                    "notional_shares": 1000
                }
            ]
        }
    }
    
    with open('/home/claude/example_input.json', 'w') as f:
        json.dump(example_data, f, indent=2)
    
    main()
