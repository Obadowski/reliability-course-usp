import numpy as np
import matplotlib.pyplot as plt


def weibull(t, alpha, beta):
    """
    Weibull distribution functions for reliability analysis.

    Parameters
    ----------
    t     : float or array-like — time (t >= 0)
    alpha : float — scale parameter (characteristic life, η); t at which ~63.2% have failed
    beta  : float — shape parameter (β); β<1: decreasing hazard, β=1: constant (exponential),
                                          β>1: increasing hazard (wear-out)

    Returns
    -------
    dict with keys:
        pdf : probability density function f(t)
        cdf : cumulative distribution function F(t) — unreliability Q(t)
        R   : reliability function R(t) = 1 - F(t)
        h   : hazard (failure) rate h(t)
    """
    t = np.asarray(t, dtype=float)

    pdf = (beta / alpha) * (t / alpha) ** (beta - 1) * np.exp(-((t / alpha) ** beta))
    cdf = 1 - np.exp(-((t / alpha) ** beta))
    R   = np.exp(-((t / alpha) ** beta))
    h   = (beta / alpha) * (t / alpha) ** (beta - 1)

    return {"pdf": pdf, "cdf": cdf, "R": R, "h": h}


def weibull_mean(alpha, beta):
    """Mean (MTTF) of the Weibull distribution."""
    return alpha * np.emath.gamma(1 + 1 / beta)


# --- Example usage ---
if __name__ == "__main__":
    # alpha = 1000   # characteristic life (e.g., hours)
    # beta  = 2.5    # shape parameter (wear-out regime)

    # t = np.linspace(0.1, 2000, 500)
    # w = weibull(t, alpha, beta)

    # fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    # fig.suptitle(f"Weibull Distribution  (α={alpha}, β={beta})")

    # axes[0, 0].plot(t, w["pdf"]);  axes[0, 0].set(title="PDF f(t)",         ylabel="f(t)")
    # axes[0, 1].plot(t, w["R"]);    axes[0, 1].set(title="Reliability R(t)", ylabel="R(t)")
    # axes[1, 0].plot(t, w["cdf"]);  axes[1, 0].set(title="Unreliability F(t)",ylabel="F(t)")
    # axes[1, 1].plot(t, w["h"]);    axes[1, 1].set(title="Hazard Rate h(t)", ylabel="h(t)")

    # for ax in axes.flat:
    #     ax.set_xlabel("t")
    #     ax.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()

    # mttf = weibull_mean(alpha, beta)
    # R_at_500 = weibull(500, alpha, beta)["R"]
    # print(f"MTTF  : {mttf:.2f}")
    # print(f"R(500): {R_at_500:.4f}")
    import sys
    t, alpha, beta = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    result = weibull(t, alpha, beta)
    for k, v in result.items():
        print(f"{k}: {v:.6f}")