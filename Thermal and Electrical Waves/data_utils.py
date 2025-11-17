def load_dataset(path):
	import pandas as pd
	import re
	
	data = pd.read_csv(path, header=3)
	timestamp = data.iloc[:, 0].to_numpy()
	output_voltage = data.iloc[:, 1].to_numpy()
	output_current = data.iloc[:, 2].to_numpy()
	thermistor_temperatures = data.iloc[:, 3:].to_numpy()

	comments = re.search(r"Comments: (.*)$", open(path).read(), re.MULTILINE)[1]

	return timestamp, output_voltage, output_current, thermistor_temperatures, comments


import numpy as np
import matplotlib.pyplot as plt

def truncate_transient(t, y, window=500, tol=0.02, plot=False, verbose=True):
    """
    Automatically remove initial transient region before steady oscillation.
    """

    t = np.asarray(t)
    y = np.asarray(y)

    # --- Center signal and compute rolling RMS amplitude ---
    y_centered = y - np.mean(y)
    roll_rms = np.sqrt(np.convolve(y_centered**2, np.ones(window)/window, mode='valid'))

    # --- Smooth RMS to avoid noise ---
    roll_rms_smooth = np.convolve(roll_rms, np.ones(window)//window, mode='same')

    # --- Detect steady-state region ---
    final_rms = np.mean(roll_rms_smooth[-window:])
    threshold = (1 + tol) * final_rms

    # Find first index where RMS stays below threshold
    idx_candidates = np.where(roll_rms_smooth < threshold)[0]
    if len(idx_candidates) == 0:
        if verbose:
            print("No stabilization detected — returning full dataset.")
        return t, y

    first_idx = idx_candidates[0]
    t_cut = t[first_idx]
    mask = t >= t_cut
    t_trunc = t[mask] - t[mask][0]
    y_trunc = y[mask]

    if verbose:
        print(f"Detected stabilization at t ≈ {t_cut:.1f}s "
              f"({first_idx} samples removed, {100*first_idx/len(t):.1f}% of data).")

    # --- Optional plot ---
    if plot:
        plt.figure(figsize=(8,4))
        plt.plot(t, y, label="Raw signal", alpha=0.6)
        plt.axvline(t_cut, color='r', linestyle='--', label=f"Cut @ {t_cut:.1f}s")
        plt.plot(t_trunc + t_cut, y_trunc, label="Truncated region", color='g')
        plt.xlabel("Time [s]")
        plt.ylabel("Temperature [°C]")
        plt.title("Transient truncation")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return t_trunc, y_trunc

