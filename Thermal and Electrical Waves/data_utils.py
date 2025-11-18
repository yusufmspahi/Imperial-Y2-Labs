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
import re

def remove_transient(t,
                     y,
                     period=None,
                     comments=None,
                     n_tail_cycles=5,
                     frac_tol=0.01,
                     min_points_per_cycle=20,
                     plot=False,
                     verbose=True):
    """
    Remove initial transient by comparing the *mean value* of each cycle.

    Algorithm
    ---------
    1. Split the data into cycles of length `period`.
    2. For each cycle k, compute the mean temperature M_k.
    3. Use the last `n_tail_cycles` to define a steady-state mean M_tail.
    4. Find the earliest cycle k_cut such that all subsequent M_k
       lie within ± frac_tol of M_tail.
    5. Cut the data at the start of cycle k_cut.

    Parameters
    ----------
    t : array-like
        Time array [s].
    y : array-like
        Temperature array, same length as t.
    period : float, optional
        Driving period τ [s]. If None, inferred from `comments` using
        a "Period = <number>" pattern.
    comments : str, optional
        Full comments string from the data file (used only if period is None).
    n_tail_cycles : int, default 5
        Number of cycles at the end used as the steady-state reference.
    frac_tol : float, default 0.01
        Fractional tolerance for the cycle mean.
        0.01 → means must be within ±1% of the tail mean.
    min_points_per_cycle : int, default 20
        Minimum number of samples required to trust a cycle.
    plot : bool, default False
        If True, plot cycle means and the chosen cut.
    verbose : bool, default True
        If True, print what cut was chosen.

    Returns
    -------
    t_trunc : np.ndarray
        Truncated time array, shifted so t_trunc[0] = 0.
    y_trunc : np.ndarray
        Truncated temperature array.
    """

    t = np.asarray(t)
    y = np.asarray(y)

    if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
        raise ValueError("t and y must be 1D arrays of the same length.")

    if t.size < min_points_per_cycle:
        if verbose:
            print("Too few points to detect transient — returning full dataset.")
        return t - t[0], y

    # --- Determine the period τ ---
    if period is None:
        if comments is None:
            raise ValueError("Either 'period' or 'comments' must be provided.")
        match = re.search(r"Period\s*=\s*([\d.]+)", comments)
        if not match:
            raise ValueError("Could not find 'Period = ...' in comments.")
        period = float(match.group(1))

    # --- Assign a cycle index to each sample ---
    cycle_index = np.floor((t - t[0]) / period).astype(int)
    n_cycles_total = cycle_index.max() + 1

    if n_cycles_total <= n_tail_cycles + 1:
        if verbose:
            print("Not enough cycles to safely detect transient — returning full dataset.")
        return t - t[0], y

    # --- Compute mean per cycle: M_k ---
    cycle_ids = []
    means = []

    for k in range(n_cycles_total):
        mask_k = (cycle_index == k)
        if np.count_nonzero(mask_k) < min_points_per_cycle:
            continue
        y_k = y[mask_k]
        M_k = np.mean(y_k)
        cycle_ids.append(k)
        means.append(M_k)

    cycle_ids = np.array(cycle_ids)
    means = np.array(means)

    if means.size <= n_tail_cycles:
        if verbose:
            print("Too few good cycles — returning full dataset.")
        return t - t[0], y

    # --- Define steady-state mean from the last n_tail_cycles ---
    tail_ids = cycle_ids[-n_tail_cycles:]
    tail_means = means[-n_tail_cycles:]
    M_tail = np.mean(tail_means)

    lower = (1.0 - frac_tol) * M_tail
    upper = (1.0 + frac_tol) * M_tail

    # Which cycles lie inside the steady band?
    in_band = (means >= lower) & (means <= upper)

    # Find earliest cycle such that *all later* cycles are within the band
    cut_cycle = None
    for idx, k in enumerate(cycle_ids):
        if np.all(in_band[idx:]):
            cut_cycle = k
            break

    if cut_cycle is None:
        if verbose:
            print("No stable mean region found — returning full dataset.")
        return t - t[0], y

    # --- Cut at the start of that cycle ---
    t_cut = t[cycle_index >= cut_cycle][0]
    mask = t >= t_cut
    t_trunc = t[mask] - t[mask][0]
    y_trunc = y[mask]

    if verbose:
        print(
            f"Detected stabilization at cycle {cut_cycle} "
            f"(t ≈ {t_cut:.1f} s, removed {np.sum(~mask)} samples, "
            f"{100*np.sum(~mask)/t.size:.1f}% of data)."
        )

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(cycle_ids, means, "o-", label="Cycle mean $M_k$")
        ax.axhline(M_tail, color="C1", linestyle="--", label="Tail mean $M_{\\mathrm{tail}}$")
        ax.fill_between(cycle_ids, lower, upper,
                        color="C1", alpha=0.2,
                        label=f"±{frac_tol*100:.2f}% band")
        ax.axvline(cut_cycle, color="r", linestyle="--",
                   label=f"Cut at cycle {cut_cycle}")
        ax.set_xlabel("Cycle index k")
        ax.set_ylabel("Mean temperature per cycle $M_k$ [°C]")
        ax.set_title("Cycle-mean based transient detection")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()

    return t_trunc, y_trunc
