import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
lookback_years = 2         # ~3 months
rolling_vol_window = 30       # days for rolling volatility
smoothing_window = 15           # for plotting

tickers = {
    "USDJPY": "JPY=X",
    "USDSGD": "SGD=X"
}

# --- Fetch data ---
data_list = []
for name, ticker in tickers.items():
    fx = yf.Ticker(ticker).history(period=f"{int(365*lookback_years)}d")['Close']
    fx.name = name
    data_list.append(fx)

data = pd.concat(data_list, axis=1)
data.index = pd.to_datetime(data.index).tz_localize(None)
data = data.asfreq('D').ffill()

# --- Compute daily log returns ---
log_returns = np.log(data / data.shift(1)).dropna()

# --- Compute rolling volatilities ---
rolling_vol = log_returns.rolling(rolling_vol_window).std() * np.sqrt(252)

# --- Normalize volatilities to [0,1] ---
normalized_vol = rolling_vol / rolling_vol.max()

# --- Initialize energies dataframe ---
energies = pd.DataFrame(index=normalized_vol.index, columns=['USD','JPY','SGD'], dtype=float)

for idx, row in normalized_vol.iterrows():
    v_usdjpy = row['USDJPY']
    v_usdsgd = row['USDSGD']

    # USD energy based on similarity
    diff = abs(np.log(np.abs(v_usdjpy - v_usdsgd)))
    usd_energy = 1 / (1 + diff)

    # Remaining energy
    residual = 1 - usd_energy

    # JPY/SGD energy proportional to their normalized volatility
    total_vol = v_usdjpy + v_usdsgd
    if total_vol > 0:
        jpy_energy = residual * (v_usdjpy / total_vol)
        sgd_energy = residual * (v_usdsgd / total_vol)
    else:
        jpy_energy = 0
        sgd_energy = 0

    # Assign to dataframe
    energies.loc[idx] = [usd_energy, jpy_energy, sgd_energy]
a
# --- Smooth for plotting ---
energies_smooth = energies.rolling(smoothing_window, min_periods=1).mean()

# --- Plot ---
plt.figure(figsize=(14,6))
colors = ['tab:red','tab:blue','tab:green']
for color, col in zip(colors, energies_smooth.columns):
    plt.plot(energies_smooth.index, energies_smooth[col], label=col)
plt.title(f"Relative Currency Energy from Normalized Rolling Volatility ({rolling_vol_window}-day window)")
plt.ylabel("Relative Energy (sum=1)")
plt.xlabel("Date")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
