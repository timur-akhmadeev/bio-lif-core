"""
Comparison of Euler method with the exact solution for a LIF neuron.

This script generates the plot used in the article:
it shows how the Euler method's error depends on the step size dt
and how it deviates from the analytical solution.

Run:
    python examples/plot_euler_vs_exact.py

Output:
    euler_vs_exact.png — plot in the project root
    Maximum Euler error at t = 20 ms is printed to the console
    to verify against the numbers in the article.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# --- Textbook neuron parameters ---
# These values are convenient for demonstration and do not claim
# to match any specific Drosophila neuron type exactly.
TAU_M = 20.0        # ms, membrane time constant
V_REST = -70.0      # mV, resting potential
R_M = 100.0         # MOhm, membrane resistance
INPUT_CURRENT = 0.1 # nA, subthreshold input current
V0 = V_REST         # mV, initial potential
DURATION = 100.0    # ms, simulation duration
V_THRESHOLD = -55.0 # mV, spike threshold (for reference)


def _check_dt(dt: float) -> float:
    """
    Validate the time step dt.

    Raises ValueError if dt <= 0, dt > DURATION, or dt is not finite.
    """
    if not np.isfinite(dt):
        raise ValueError(f"dt must be a finite number, got {dt}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if dt > DURATION:
        raise ValueError(
            f"dt must not exceed the simulation duration "
            f"({DURATION} ms), got {dt}"
        )
    return dt


def exact_solution(t: np.ndarray) -> np.ndarray:
    """
    Analytical LIF solution for a constant input current.

    V(t) = V_inf + (V0 - V_inf) * exp(-t / tau_m)

    where V_inf = V_rest + R_m * I.
    """
    v_inf = V_REST + R_M * INPUT_CURRENT
    return v_inf + (V0 - v_inf) * np.exp(-t / TAU_M)


def euler_solution(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerical LIF solution using the Euler method.

    V_{n+1} = V_n + dt/tau_m * [-(V_n - V_rest) + R_m * I]

    Returns (times, voltages).
    """
    _check_dt(dt)

    n_steps = int(round(DURATION / dt))
    times = np.zeros(n_steps + 1)
    voltages = np.zeros(n_steps + 1)
    voltages[0] = V0

    v = V0
    for n in range(n_steps):
        dv = (
            -(v - V_REST) + R_M * INPUT_CURRENT
        ) / TAU_M * dt
        v += dv
        times[n + 1] = (n + 1) * dt
        voltages[n + 1] = v

    return times, voltages


def _error_at_t(times: np.ndarray, voltages: np.ndarray, t_target: float) -> float:
    """
    Absolute error between Euler and exact solution at the closest
    sample to t_target.
    """
    idx = int(np.argmin(np.abs(times - t_target)))
    v_exact_at_t = exact_solution(np.array([times[idx]]))[0]
    return float(abs(voltages[idx] - v_exact_at_t))


def main() -> None:
    # --- Exact solution on a dense grid (smooth curve) ---
    t_exact = np.linspace(0, DURATION, 1000)
    v_exact = exact_solution(t_exact)

    # --- Euler method with two different step sizes ---
    times_1, v_euler_1 = euler_solution(dt=1.0)
    times_5, v_euler_5 = euler_solution(dt=5.0)

    # --- Sanity check: exact solution vs closed-form ---
    v_inf = V_REST + R_M * INPUT_CURRENT
    v_exact_check = v_inf + (V0 - v_inf) * np.exp(-t_exact / TAU_M)
    assert np.allclose(v_exact, v_exact_check), "Analytical formula mismatch"

    # --- Sanity check: Euler vs closed-form for linear ODE ---
    # For a linear ODE, Euler has a closed form:
    # V_{n+1} = V_inf + (1 - dt/tau_m) * (V_n - V_inf)
    decay_euler = 1.0 - 1.0 / TAU_M
    v_check = np.zeros_like(v_euler_1)
    v_check[0] = V0
    for n in range(len(v_check) - 1):
        v_check[n + 1] = v_inf + decay_euler * (v_check[n] - v_inf)
    assert np.allclose(v_euler_1, v_check, atol=1e-13), "Euler formula mismatch"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    # Exact solution — black dashed line (zorder=3 to keep it on top)
    ax.plot(
        t_exact, v_exact,
        "k--", linewidth=2, zorder=3,
        label="Exact solution",
    )

    # Euler method, dt = 5 ms — orange with markers
    ax.plot(
        times_5, v_euler_5,
        color="tab:orange", linewidth=1.5, zorder=2,
        marker="o", markersize=5,
        label="Euler, dt = 5 ms",
    )

    # Euler method, dt = 1 ms — green
    ax.plot(
        times_1, v_euler_1,
        color="tab:green", linewidth=1.5, zorder=1,
        label="Euler, dt = 1 ms",
    )

    # Axis labels
    ax.set_xlabel("Time, ms")
    ax.set_ylabel("Membrane potential, mV")

    # Title with parameters (indices in italics via mathtext)
    title = (
        "LIF: Euler method vs exact solution\n"
        f"$\\tau_m = {TAU_M:.0f}$ ms, "
        f"$V_{{rest}} = {V_REST:.0f}$ mV, "
        f"$R_m = {R_M:.0f}$ MOhm, "
        f"$I = {INPUT_CURRENT}$ nA, "
        f"$V_0 = {V0:.0f}$ mV, "
        f"$V_{{threshold}} = {V_THRESHOLD:.0f}$ mV"
    )
    ax.set_title(title, fontsize=11)

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to project root
    out_path = Path(__file__).parent.parent / "euler_vs_exact.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")

    # Print Euler error at t = 20 ms to verify against the article
    t_check = 20.0
    err_1 = _error_at_t(times_1, v_euler_1, t_check)
    err_5 = _error_at_t(times_5, v_euler_5, t_check)
    print(f"Maximum Euler error (t = {t_check:.0f} ms):")
    print(f"  dt = 1 ms: {err_1:.3f} mV")
    print(f"  dt = 5 ms: {err_5:.3f} mV")

    plt.show()


if __name__ == "__main__":
    main()