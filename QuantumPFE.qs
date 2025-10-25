namespace QuantumPFE {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Arithmetic;
    open Microsoft.Quantum.Preparation;
    open Microsoft.Quantum.Characterization;
    open Microsoft.Quantum.Oracles;
    open Microsoft.Quantum.AmplitudeAmplification;

    /// # Summary
    /// Prepares a quantum state representing a normal distribution for asset prices
    /// 
    /// # Input
    /// ## mu
    /// Drift parameter for the asset
    /// ## sigma 
    /// Volatility parameter for the asset
    /// ## time
    /// Time horizon for the simulation
    /// ## register
    /// Qubit register to encode the distribution
    operation PrepareNormalDistribution(
        mu : Double,
        sigma : Double, 
        time : Double,
        register : Qubit[]
    ) : Unit is Adj + Ctl {
        let nQubits = Length(register);
        let nStates = PowI(2, nQubits);
        
        // Calculate the discretized normal distribution amplitudes
        mutable amplitudes = [0.0, size = nStates];
        let sqrtTime = Sqrt(time);
        let variance = sigma * sqrtTime;
        
        // Create discretized price levels
        let minPrice = mu - 5.0 * variance;
        let maxPrice = mu + 5.0 * variance;
        let priceStep = (maxPrice - minPrice) / IntAsDouble(nStates - 1);
        
        mutable normalizationFactor = 0.0;
        
        for i in 0..nStates-1 {
            let price = minPrice + IntAsDouble(i) * priceStep;
            let exponent = -0.5 * PowD((price - mu) / variance, 2.0);
            set amplitudes w/= i <- ExpD(exponent);
            set normalizationFactor += PowD(amplitudes[i], 2.0);
        }
        
        // Normalize the amplitudes
        set normalizationFactor = Sqrt(normalizationFactor);
        for i in 0..nStates-1 {
            set amplitudes w/= i <- amplitudes[i] / normalizationFactor;
        }
        
        // Prepare the quantum state
        PrepareArbitraryStateD(amplitudes, LittleEndian(register));
    }

    /// # Summary
    /// Computes the payoff for a European option
    ///
    /// # Input
    /// ## optionType
    /// Type of option: 0 for call, 1 for put
    /// ## strike
    /// Strike price of the option
    /// ## spot
    /// Current spot price
    /// ## priceRegister
    /// Qubit register encoding the asset price
    /// ## payoffRegister
    /// Qubit register to store the payoff
    operation ComputeEuropeanOptionPayoff(
        optionType : Int,
        strike : Double,
        spot : Double,
        priceRegister : Qubit[],
        payoffRegister : Qubit[]
    ) : Unit is Adj + Ctl {
        let nPriceBits = Length(priceRegister);
        let nPayoffBits = Length(payoffRegister);
        
        // Create oracle for payoff computation
        within {
            // Prepare comparison circuit
            let strikeInt = Round(strike * PowD(2.0, IntAsDouble(nPriceBits)) / (spot * 2.0));
            
            using ancilla = Qubit() {
                // Compare price with strike
                GreaterThanOrEqualI(LittleEndian(priceRegister), strikeInt, ancilla);
                
                if optionType == 0 { // Call option
                    // Payoff = max(S - K, 0)
                    Controlled ApplyPayoffTransform([ancilla], (priceRegister, payoffRegister, strike, spot, true));
                } else { // Put option  
                    // Payoff = max(K - S, 0)
                    X(ancilla); // Flip for put option condition
                    Controlled ApplyPayoffTransform([ancilla], (priceRegister, payoffRegister, strike, spot, false));
                    X(ancilla); // Uncompute
                }
            }
        } apply {
            // Payoff has been computed and stored in payoffRegister
        }
    }

    /// # Summary
    /// Helper operation to apply payoff transformation
    operation ApplyPayoffTransform(
        priceRegister : Qubit[],
        payoffRegister : Qubit[],
        strike : Double,
        spot : Double,
        isCall : Bool
    ) : Unit is Adj + Ctl {
        // Simplified payoff encoding
        let nBits = Length(priceRegister);
        
        for i in 0..Length(payoffRegister)-1 {
            if i < nBits {
                if isCall {
                    CNOT(priceRegister[i], payoffRegister[i]);
                } else {
                    X(payoffRegister[i]); // For put, invert the bits
                    CNOT(priceRegister[i], payoffRegister[i]);
                }
            }
        }
    }

    /// # Summary
    /// Quantum Amplitude Estimation for PFE calculation
    ///
    /// # Input
    /// ## portfolio
    /// Portfolio data structure
    /// ## confidence
    /// Confidence level for PFE (e.g., 0.95 for 95%)
    /// ## nIterations
    /// Number of iterations for amplitude estimation
    operation QuantumAmplitudeEstimationPFE(
        portfolioSize : Int,
        confidence : Double,
        nIterations : Int
    ) : Double {
        mutable estimate = 0.0;
        
        let nPriceQubits = 4; // Simplified for initial implementation
        let nAncillaQubits = 3;
        
        use qubits = Qubit[portfolioSize * nPriceQubits + nAncillaQubits] {
            // State preparation oracle
            let statePrep = PreparePortfolioState(portfolioSize, nPriceQubits, _, _);
            
            // Grover oracle for threshold detection
            let groverOracle = ReflectAboutMarked(confidence, _);
            
            // Run amplitude estimation
            let phases = EstimatePhases(
                nIterations,
                statePrep,
                groverOracle,
                qubits[0..portfolioSize * nPriceQubits - 1],
                qubits[portfolioSize * nPriceQubits..]
            );
            
            // Convert phase to probability estimate
            set estimate = ConvertPhaseToAmplitude(phases[0]);
            
            ResetAll(qubits);
        }
        
        return estimate;
    }

    /// # Summary
    /// Prepares the quantum state for the entire portfolio
    operation PreparePortfolioState(
        portfolioSize : Int,
        qubitsPerAsset : Int,
        stateQubits : Qubit[],
        ancillaQubits : Qubit[]
    ) : Unit is Adj + Ctl {
        // Initialize superposition
        for i in 0..portfolioSize-1 {
            let assetQubits = stateQubits[i * qubitsPerAsset..(i + 1) * qubitsPerAsset - 1];
            
            // Prepare normal distribution for each asset
            // Using simplified parameters for demonstration
            PrepareNormalDistribution(
                0.02,  // drift (r)
                0.20,  // volatility (example)
                0.25,  // time horizon
                assetQubits
            );
        }
    }

    /// # Summary
    /// Reflects about marked states above threshold
    operation ReflectAboutMarked(threshold : Double, register : Qubit[]) : Unit is Adj + Ctl {
        // Mark states where portfolio value exceeds threshold
        within {
            // This is a simplified marking oracle
            ApplyToEachCA(Z, register[0..2]);
        } apply {
            // Reflection
            ApplyToEachCA(X, register);
            Controlled Z(Most(register), Tail(register));
            ApplyToEachCA(X, register);
        }
    }

    /// # Summary
    /// Estimate phases using quantum phase estimation
    operation EstimatePhases(
        nIterations : Int,
        statePreparation : ((Qubit[], Qubit[]) => Unit is Adj + Ctl),
        oracle : (Qubit[] => Unit is Adj + Ctl),
        targetQubits : Qubit[],
        ancillaQubits : Qubit[]
    ) : Double[] {
        mutable phases = [0.0, size = nIterations];
        
        for iter in 0..nIterations-1 {
            // Prepare initial state
            statePreparation(targetQubits, ancillaQubits);
            
            // Apply Grover operator multiple times
            let nGrover = PowI(2, iter);
            for _ in 0..nGrover-1 {
                oracle(targetQubits);
                ReflectAboutUniform(targetQubits);
            }
            
            // Measure and estimate phase
            let measurement = MeasureInteger(LittleEndian(ancillaQubits));
            set phases w/= iter <- IntAsDouble(measurement) / IntAsDouble(PowI(2, Length(ancillaQubits)));
            
            ResetAll(targetQubits + ancillaQubits);
        }
        
        return phases;
    }

    /// # Summary
    /// Reflects about uniform superposition
    operation ReflectAboutUniform(register : Qubit[]) : Unit is Adj + Ctl {
        within {
            ApplyToEachCA(H, register);
            ApplyToEachCA(X, register);
        } apply {
            Controlled Z(Most(register), Tail(register));
        }
    }

    /// # Summary
    /// Converts phase estimation to amplitude
    function ConvertPhaseToAmplitude(phase : Double) : Double {
        return PowD(Sin(PI() * phase), 2.0);
    }

    /// # Summary
    /// Greater than or equal comparison for integers
    operation GreaterThanOrEqualI(
        value : LittleEndian,
        threshold : Int,
        result : Qubit
    ) : Unit is Adj + Ctl {
        // Simplified comparison - marks if MSB is set
        CNOT(value![Length(value!) - 1], result);
    }

    /// # Summary
    /// Main entry point for PFE calculation
    @EntryPoint()
    operation CalculatePFE() : Double {
        Message("Starting Quantum PFE Calculation...");
        
        // Parameters from the portfolio
        let portfolioSize = 7; // Number of positions
        let confidence = 0.95; // 95% confidence level
        let nIterations = 3; // Number of QAE iterations
        
        let pfe = QuantumAmplitudeEstimationPFE(portfolioSize, confidence, nIterations);
        
        Message($"Estimated PFE at {confidence * 100.0}% confidence: {pfe}");
        
        return pfe;
    }
}
