import numpy as np
import pandas as pd
import os

# ==========================================
# Settings
# ==========================================

OUTPUT_DIR = "mock_data"

periods = [10.0, 15.0, 20.0, 30.0, 60.0]   # periods τ in seconds
N_RUNS = 3
N_THERM = 8
DELTA_D_MM = 5.0                    # thermistor spacing (mm)

D_TRUE = 35.0                       # thermal diffusivity in mm^2/s
DURATION = 600.0                    # total duration in seconds
FS = 10.0                           # sampling frequency (Hz)
NOISE_STD = 0.1                     # realistic noise
T0 = 20.0                           # baseline temp
A0 = 2.0                            # amplitude at thermistor 0


# ==========================================
# Helper: analytic plane-wave temperature
# ==========================================

def thermal_wave_model(t, tau, x_mm):
    """Generate thermal wave time series at position x_mm for period τ."""
    f = 1.0 / tau
    omega = 2 * np.pi * f
    k = np.sqrt(omega / (2 * D_TRUE))  # units: 1/mm

    A = A0 * np.exp(-k * x_mm)
    phi = -k * x_mm

    y = T0 + A * np.sin(omega * t + phi)
    y += NOISE_STD * np.random.randn(len(t))
    return y


# ==========================================
# Main generator
# ==========================================

def generate_mock_task16():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t = np.arange(0, DURATION, 1.0 / FS)
    x_positions = np.arange(N_THERM) * DELTA_D_MM

    files_created = []

    for tau in periods:
        for run in range(1, N_RUNS + 1):

            # Build table
            data = {
                "timestamp": t,
                "output_voltage": np.zeros_like(t),
                "output_current": np.zeros_like(t),
            }

            for i in range(N_THERM):
                data[f"therm_{i}"] = thermal_wave_model(t, tau, x_positions[i])

            df = pd.DataFrame(data)

            # Filename
            filename = f"thermal_mock_T{int(tau)}s_run{run}.csv"
            path = os.path.join(OUTPUT_DIR, filename)

            # Write file in the exact format required by load_dataset
            with open(path, "w") as f_out:
                f_out.write("Dummy line 1\n")
                f_out.write("Dummy line 2\n")
                f_out.write(f"Comments: Period = {tau}\n")  # <-- period instead of freq
                df.to_csv(f_out, index=False)

            files_created.append(path)

    print("Created files:")
    for f in files_created:
        print("  ", f)


# Run it
generate_mock_task16()
