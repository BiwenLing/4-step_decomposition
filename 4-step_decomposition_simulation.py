#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, mvn, multivariate_normal
import matplotlib.ticker as mtick
import scipy.stats as st
from scipy.optimize import minimize_scalar

def E_Y1(params):
    """Closed‐form E_P[Y1]."""
    y0, mu, T = params["y0"], params["mu"], params["T"]
    return y0 * np.exp(mu * T)

def Var_Y1(params):
    """Closed‐form Var_P[Y1]."""
    y0, mu, sigma, T = params["y0"], params["mu"], params["sigma"], params["T"]
    return (y0**2) * np.exp(2*mu*T) * (np.exp(sigma**2 * T)-1)

def biv_cdf(u, v, rho):
    """Bivariate standard normal CDF with corr=rho at (u, v)."""
    # scipy's multivariate_normal.cdf takes an array of points and returns array
    return multivariate_normal.cdf([u, v], mean=[0, 0], cov=[[1, rho], [rho, 1]])

def E_S(params):
    """
    Closed-form E_P[S] matching equation (3.33).
    """
    N        = params["N"]
    y0       = params["y0"]
    mu_f     = params["mu"]
    sigma_f  = params["sigma"]
    T        = params["T"]
    mu_s     = params["mu_s"]
    sigma_s  = params["sigma_s"]
    rho      = params["rho"]
    K        = params["K"]
    alpha    = params["alpha"]
    r        = params["r"]

    # compute standardized constants
    c = mu_s / sigma_s
    d1 = (np.log(y0/K) + (mu_f + 0.5*sigma_f**2)*T) / (sigma_f * np.sqrt(T))
    d2 = d1 - sigma_f * np.sqrt(T)

    # term1
    term1 = N * np.exp(mu_s + 0.5*sigma_s**2) * (
        norm.cdf(-c - sigma_s) 
        - alpha * K * biv_cdf(d2 + rho*sigma_s, -c - sigma_s, -rho)
    )
    # term2
    term2 = N * alpha * y0 * np.exp(mu_f*T + mu_s + 0.5*sigma_s**2 + rho*sigma_s*sigma_f*np.sqrt(T)) * \
        biv_cdf(d1 + rho*sigma_s,
                -c - rho*sigma_f*np.sqrt(T) - sigma_s,
                -rho)
    # term3
    term3 = N * (norm.cdf(c) - alpha * K * biv_cdf(d2, c, rho))
    # term4
    term4 = N * alpha * y0 * np.exp(mu_f*T) * biv_cdf(d1, c + rho*sigma_f*np.sqrt(T), rho)

    return term1 + term2 + term3 + term4

def E_SY1(params):
    """
    Closed-form E_P[S * Y1] matching equation (3.32).
    """
    N        = params["N"]
    y0       = params["y0"]
    mu_f     = params["mu"]
    sigma_f  = params["sigma"]
    T        = params["T"]
    mu_s     = params["mu_s"]
    sigma_s  = params["sigma_s"]
    rho      = params["rho"]
    K        = params["K"]
    alpha    = params["alpha"]

    c = - mu_s / sigma_s
    d1 = (np.log(y0/K) + (mu_f + 0.5*sigma_f**2)*T) / (sigma_f * np.sqrt(T))

    a1 = np.exp(mu_f*T + mu_s + 0.5*sigma_s**2 + rho*sigma_s*sigma_f*np.sqrt(T))
    a2 = np.exp(2*mu_f*T + sigma_f**2*T + mu_s + 0.5*sigma_s**2 + 2*rho*sigma_s*sigma_f*np.sqrt(T))
    b1 = np.exp(mu_f*T)
    b2 = np.exp(2*mu_f*T + sigma_f**2*T)

    # term A
    A = N * y0 * a1 * norm.cdf(c - sigma_s - rho*sigma_f*np.sqrt(T))
    # term B
    B = - N * alpha * K * y0 * a1 * biv_cdf(d1 + rho*sigma_s, c - sigma_s - rho*sigma_f*np.sqrt(T), -rho)
    # term C
    C = N * alpha * y0**2 * a2 * biv_cdf(d1 + sigma_f*np.sqrt(T) + rho*sigma_s,c - sigma_s - 2*rho*sigma_f*np.sqrt(T), -rho)
    # term D
    D = N * y0 * b1 * (norm.cdf(-c + rho*sigma_f*np.sqrt(T))
        - alpha * K * biv_cdf(d1, -c + rho*sigma_f*np.sqrt(T), rho))
    # term E
    E = N * alpha * y0**2 * b2 * biv_cdf(d1 + sigma_f*np.sqrt(T), -c + 2*rho*sigma_f*np.sqrt(T), rho)

    return A + B + C + D + E

def simulate_decomp(params, n_sim=100000, seed=None):
    """
    Simulate the static one-period decomposition, but compute
    theta0 and theta1 from closed-form E[], Var[], Cov[].
    Returns a dict with keys 'theta0','theta1','S','Y_h','Y_s_fin','Y_s_res','Y_i'.
    """
    if seed is not None:
        np.random.seed(seed)

    # Unpack parameters
    N       = params["N"]
    mu      = params["mu"]
    sigma   = params["sigma"]
    T       = params["T"]
    mu_s    = params["mu_s"]
    sigma_s = params["sigma_s"]
    rho     = params["rho"]
    K       = params["K"]
    alpha   = params["alpha"]
    y0      = params.get("y0", 1.0)
    r       = params["r"]

    # 1) draw underlying normals
    Y = np.random.randn(n_sim)
    m_y = mu_s + rho * sigma_s * Y
    s_y = sigma_s * np.sqrt(1 - rho**2)
    bar_Z = m_y + s_y * np.random.randn(n_sim)
    Z = np.where(bar_Z < 0, bar_Z, 0.0)

    # 2) terminal stock marginal under P
    Y1 = y0 * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Y)

    # 3) Bernoulli draws
    p = np.clip(np.exp(Z), 0.0, 1.0)
    X = np.random.binomial(1, p[:, None], size=(n_sim, N))
    Xsum = X.sum(axis=1)

    # 4) total payoff
    payoff_factor = 1 + alpha * np.maximum(Y1 - K, 0.0)
    S = payoff_factor * Xsum

    # --------------------------------------------------------------------
    # 5) closed-form theta's
    EY1   = E_Y1(params)
    VarY1 = Var_Y1(params)
    ES    = E_S(params)
    ESY1  = E_SY1(params)

    CovSY1 = ESY1 - ES * EY1
    theta1 = CovSY1 / VarY1
    theta0 = (ES - theta1 * EY1) * np.exp(-r * T)
    # --------------------------------------------------------------------

    # 6) analytic E[X1 | Y]
    I1 = norm.cdf(m_y / s_y)
    I2 = np.exp(m_y + 0.5*s_y**2) * norm.cdf(-m_y/s_y - s_y)
    EX1_given_Y = I1 + I2

    # 7) decompose
    Y_h     = theta0 * np.exp(r * T)+ theta1 * Y1
    Y_s_fin = N * payoff_factor * EX1_given_Y - Y_h
    Y_s_act = N * payoff_factor * (np.exp(Z) - EX1_given_Y)
    Y_i     = S - (Y_h + Y_s_fin + Y_s_act)

    return {
        "theta0":   theta0,
        "theta1":   theta1,
        "S":        S,
        "Y_h":      Y_h,
        "Y_s_fin":  Y_s_fin,
        "Y_s_act":  Y_s_act,
        "Y_i":      Y_i,
    }



def plot_decomp(decomp,
                n_bars=50,
                exclude_extremes=True,
                extreme_pct=0.01,
                save_dir=None):
    """
    Plot stacked bar of the first n_bars simulation paths.
    If exclude_extremes, drop paths where any component is in
      the top or bottom `extreme_pct` quantile.
    Optionally save to PDF if save_dir is provided.
    """
    # Extract arrays
    S       = decomp["S"]
    comps   = [decomp[k] for k in ("Y_h","Y_s_fin","Y_s_act","Y_i")]
    labels  = [r"$Y^h$", r"$Y^s_{fin}$", r"$Y^s_{act}$", r"$Y^i$"]
    colors = [
    "#ADD8E6",  # LightBlue
    "#FF4500",  # OrangeRed
    "#228B22",  # ForestGreen
    "#800080",  # Purple
    ]

    n_sim   = len(S)

    # Build a mask to exclude extremes if requested
    mask = np.ones(n_sim, dtype=bool)
    if exclude_extremes:
        for arr in comps + [S]:
            low, high = np.quantile(arr, extreme_pct), np.quantile(arr, 1-extreme_pct)
            mask &= (arr >= low) & (arr <= high)

    # Select the first n_bars surviving paths
    idxs_all = np.nonzero(mask)[0]
    if len(idxs_all) < n_bars:
        raise ValueError(f"Not enough non-extreme paths: only {len(idxs_all)} remain.")
    chosen = idxs_all[:n_bars]

    # Prepare positive/negative bottoms
    bottom_pos = np.zeros(n_bars)
    bottom_neg = np.zeros(n_bars)

    # Create figure + axes
    fig, ax = plt.subplots(figsize=(12,6))

    # Plot each component
    for arr, lbl, col in zip(comps, labels, colors):
        vals = arr[chosen]
        pos  = np.clip(vals, 0, None)
        neg  = np.clip(vals, None, 0)

        ax.bar(np.arange(n_bars), pos,
               bottom=bottom_pos, color=col, width=0.8, label=lbl)
        bottom_pos += pos

        ax.bar(np.arange(n_bars), neg,
               bottom=bottom_neg, color=col, width=0.8)
        bottom_neg += neg

    # Zero line
    ax.axhline(0, color="black", linewidth=0.8)

    # Scale y-limits
    ax.set_ylim(-200, 1000)

    # Labels
    ax.set_xlabel("Simulation Path", fontsize=12)
    ax.set_ylabel("Contribution to S", fontsize=12)

    # Legend outside
    ax.legend(loc="upper left",
              bbox_to_anchor=(1.02, 1),
              borderaxespad=0,
              fontsize="medium")

    # Tidy layout, reserving room on the right for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"stacked_decomp_n{n_bars}.pdf"
        fig.savefig(os.path.join(save_dir, filename), bbox_inches="tight")

    plt.show()


# In[7]:


mu_1 = 0.113826
sigma_1 = 0.002990
lambda_0 = 0.015030
T = 10
mu_s = -lambda_0 / mu_1 * (np.exp(mu_1 * T) - 1)
sigma_s = (sigma_1**2 / (2 * mu_1**3) * (np.exp(mu_1* 2 * T) - 4 * np.exp(mu_1 * T) + 3 + 2 * mu_1 * T))**0.5
print(mu_s)
print(sigma_s)

params = {
    "N":         500,
    "mu":        0.03,
    "sigma":     0.25,
    "T":         10.0,
    "mu_s":      mu_s,
    "sigma_s":   sigma_s,
    "rho":       0,
    "K":         1,
    "alpha":     0.5,
    "y0":        0.7,
    "r":         0.02
}

# 1) simulate once (with a fixed seed)
decomp = simulate_decomp(params, n_sim=100000, seed=42)

#print(np.mean(decomp['S']), np.mean(decomp['Y_h']), np.mean(decomp['Y_s_fin']), np.mean(decomp['Y_s_act']), np.mean(decomp['Y_i']))

# 2) plot, dropping the top/bottom 1% of extreme paths
plot_decomp(decomp,
                n_bars=50,
                exclude_extremes=True,
                extreme_pct=0.01,
                save_dir = "/Users/lingbiwen/Desktop/Decomposition/Pics")


# In[8]:


print(np.var(decomp['S']), np.var(decomp['Y_h']) + np.var(decomp['Y_s_fin']) + np.var(decomp['Y_s_act'])+ np.var(decomp['Y_i']))
print(E_S(params))


# In[10]:


def plot_3parts_decomp(params,
                       n_sim = 10000,
                       seed = 42,
                       n_bars=50,
                       save_dir=None):
    """
    Plot stacked bar of the non-hedgeable part of the first n_bars simulation paths.
    If exclude_extremes, drop paths where any component is in
      the top or bottom `extreme_pct` quantile.
    Optionally save to PDF if save_dir is provided.
    """
    # Extract arrays
    rho = params["rho"]
    decomp = simulate_decomp(params, n_sim, seed)
    comps   = [decomp[k] for k in ("Y_s_fin","Y_s_act","Y_i")]
    labels  = [r"$Y^s_{fin}$", r"$Y^s_{act}$", r"$Y^i$"]
    colors = [
    "#FF4500",  # OrangeRed
    "#228B22",  # ForestGreen
    "#800080",  # Purple
    ]
    n_sim   = len(decomp["Y_i"])

    # Build a mask 
    mask = np.ones(n_sim, dtype=bool)

    # Select the first n_bars surviving paths
    idxs_all = np.nonzero(mask)[0]
    if len(idxs_all) < n_bars:
        raise ValueError(f"Not enough non-extreme paths: only {len(idxs_all)} remain.")
    chosen = idxs_all[:n_bars]

    # Prepare positive/negative bottoms
    bottom_pos = np.zeros(n_bars)
    bottom_neg = np.zeros(n_bars)

    # Create figure + axes
    fig, ax = plt.subplots(figsize=(12,6))

    # Plot each component
    for arr, lbl, col in zip(comps, labels, colors):
        vals = arr[chosen]
        pos  = np.clip(vals, 0, None)
        neg  = np.clip(vals, None, 0)

        ax.bar(np.arange(n_bars), pos,
               bottom=bottom_pos, color=col, width=0.8, label=lbl)
        bottom_pos += pos

        ax.bar(np.arange(n_bars), neg,
               bottom=bottom_neg, color=col, width=0.8)
        bottom_neg += neg

    # Zero line
    ax.axhline(0, color="black", linewidth=0.8)

    # Autoscale y-limits with 5% margin
    allbars = np.concatenate([bottom_pos, bottom_neg])
    lo, hi  = allbars.min(), allbars.max()
    margin  = 0.05 * (hi - lo)
    ax.set_ylim(-150, 150)

    # Labels
    ax.set_xlabel("Simulation Path", fontsize=12)
    ax.set_ylabel("Contribution to S", fontsize=12)

    # Legend outside
    ax.legend(loc="upper left",
              bbox_to_anchor=(1.02, 1),
              borderaxespad=0,
              fontsize="medium")

    # Tidy layout, reserving room on the right for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"stacked_decomp_3parts_n{n_bars}_rho{rho}.pdf"
        fig.savefig(os.path.join(save_dir, filename), bbox_inches="tight")

    plt.show()



# Plot the decomposition for the non-hedgeable part that is composed of the non-hedgeable hedgeable part 
# Y^i, Y^s_{fin}, and Y^s_{res}
rho_lists = [-0.8, -0.4, 0, 0.4, 0.8]
for rho in rho_lists:
    p = params.copy()
    p["rho"] = rho
    plot_3parts_decomp(p,
                       n_sim = 10000,
                       seed = 42,
                       n_bars=50,
                       save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")




# In[11]:


def var_Yh_vs_rho(params, rhos):
    """Analytic var(Y_h) over an array of rho."""
    varY1 = Var_Y1(params)
    out = []
    for rho in rhos:
        p = params.copy(); p["rho"] = rho
        ES   = E_S(p)
        ESY1 = E_SY1(p)
        EY1  = E_Y1(p)
        theta1 = (ESY1 - ES*EY1)/varY1
        out.append(theta1**2 * varY1)
    return np.array(out)


def var_vs_rho(params, rhos, n_sim=100_000, seed=42):
    """Monte Carlo var(S), var(Y^s_act), var(Y^i), var(Y^s_fin) over an array of rho."""
    out_var_Ys_act = []
    out_var_Ys_fin = []
    out_var_Yi = []
    out_var_S = []
    for rho in rhos:
        p = params.copy(); p["rho"] = rho
        sim_results = simulate_decomp(p, n_sim = n_sim, seed = seed)
        Y_s_act = sim_results['Y_s_act']
        Y_s_fin = sim_results['Y_s_fin']
        Y_i = sim_results['Y_i']
        S = sim_results['S']
        out_var_Ys_act.append(np.var(Y_s_act))
        out_var_Ys_fin.append(np.var(Y_s_fin))
        out_var_Yi.append(np.var(Y_i))
        out_var_S.append(np.var(S))
    return np.array(out_var_Ys_fin), np.array(out_var_Ys_act), np.array(out_var_Yi), np.array(out_var_S)


# ——— Plotting wrapper ———

def plot_all_vars_vs_rho(params,
                         save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    os.makedirs(save_dir, exist_ok=True)
    rhos = np.linspace(-0.99, 0.99, 100)
    N = params["N"]   

    var_vs_rho_results = var_vs_rho(params, rhos, n_sim=100_000, seed=42)
    vFin = var_vs_rho_results[0]/(N**2)
    vAct = var_vs_rho_results[1]/(N**2)
    vIdio = var_vs_rho_results[2]/(N**2)

    plt.figure(figsize=(8,5))
    plt.plot(rhos, vFin, label=r"$\frac{\mathrm{Var}(Y^s_{fin})}{N^2}$", lw=2, color = "#FF4500")
    plt.plot(rhos, vAct, label=r"$\frac{\mathrm{Var}(Y^s_{act})}{N^2}$", lw=2, color = "#228B22")
    plt.plot(rhos, vIdio, label=r"$\frac{\mathrm{Var}(Y^{i})}{N^2}$", lw=2, color = "#800080")
    plt.xlabel("Correlation $\\rho$")
    plt.ylabel("Variance")
    plt.xlim(-1,1)
    plt.ylim(-0.0005,0.010)
    #plt.title("Component Variances vs Correlation $\\rho$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/vars_components_vs_rho.pdf", bbox_inches="tight")
    plt.show()

params = {
    "N":         500,
    "mu":        0.03,
    "sigma":     0.25,
    "T":         10.0,
    "mu_s":      mu_s,
    "sigma_s":   sigma_s,
    "rho":       0,
    "K":         1,
    "alpha":     0.5,
    "y0":        0.7,
    "r":         0.02
}

plot_all_vars_vs_rho(params)


# In[15]:


def plot_hedging_unit_stock_vs_rho(params, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    N = params["N"]
    r = params["r"]
    T = params["T"]
    rhos = np.linspace(-0.99, 0.99, 100)
    varY1 = Var_Y1(params)
    theta1_per_policy_vs_rho = []
    theta0_per_policy_vs_rho = []
    for rho in rhos:
        p = params.copy(); p["rho"] = rho
        ES   = E_S(p)
        ESY1 = E_SY1(p)
        EY1  = E_Y1(p)
        theta1_per_policy = 1/N * (ESY1 - ES*EY1)/varY1
        theta0_per_policy = (ES/N - theta1_per_policy* EY1) * np.exp(-r * T)
        theta1_per_policy_vs_rho.append(theta1_per_policy)
        theta0_per_policy_vs_rho.append(theta0_per_policy)

    plt.figure(figsize=(8,5))
    plt.plot(rhos, theta1_per_policy_vs_rho, label=r"$\theta^{(1)}$", lw=2)
    plt.plot(rhos, theta0_per_policy_vs_rho, label=r"$\theta^{(0)}$", lw=2)
    plt.xlabel("Correlation $\\rho$")
    plt.xlim(-1,1)
    plt.ylabel("Unit")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/theta1_vs_rho.pdf", bbox_inches="tight")
    plt.show()



def plot_hist_Yi_vs_N(params, n_list, num_sim=50000, seed= 42, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    """
    Plot histogram of Y_i/n (residual) for each n in rho_list.
    The y-axis shows relative frequency (density = count / num_sim).
    Optionally save each figure as 'hist_Yi_n{n}.pdf' in save_dir.
    """
    for n in n_list:
        p = params.copy(); p["N"] = n
        Y_i_over_N = simulate_decomp(p, n_sim=50000, seed= seed)['Y_i']/n

        fig, ax = plt.subplots(figsize=(6, 5))
        weights = np.ones_like(Y_i_over_N) / num_sim
        ax.hist(Y_i_over_N, bins=50, weights=weights, edgecolor='black')
        ax.set_xlim(-0.30, 0.30)
        ax.set_ylim(0, 0.30)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        #ax.set_title(fr"Histogram of $Y^i$ for $N={n}$")
        ax.set_xlabel(r"$\frac{Y^i}{N}$")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            fig.savefig(f"{save_dir}/hist_Yi_n{n}.pdf", format='pdf', bbox_inches='tight')
        plt.show()
        plt.close(fig)

plot_hedging_unit_stock_vs_rho(params, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")


# In[188]:


params = {
    "N":         500,
    "mu":        0.05,
    "sigma":     0.25,
    "T":         10.0,
    "mu_s":      mu_s,
    "sigma_s":   sigma_s,
    "rho":       0,
    "K":         1,
    "alpha":     0.1,
    "y0":        0.7,
    "r":         0.02
}

n_list = [100, 2000]
plot_hist_Yi_vs_N(params, n_list, num_sim=50000, seed= 42, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")
plot_hedging_unit_stock_vs_rho(params, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")


# In[11]:


def plot_hist_Yh_vs_rho(params, rho_list, num_sim=50000, seed= 42, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    """
    Plot histogram of Y_h (residual) for each rho in rho_list.
    The y-axis shows relative frequency (density = count / num_sim).
    Optionally save each figure as 'hist_Yh_rho{rho}.pdf' in save_dir.
    """
    for rho in rho_list:
        N = params["N"]
        p = params.copy(); p["rho"] = rho
        Y_h = simulate_decomp(p, n_sim=50000, seed= seed)['Y_h']


        fig, ax = plt.subplots(figsize=(6, 5))
        weights = np.ones_like(Y_h) / num_sim
        ax.hist(Y_h, bins=50, weights=weights, edgecolor='black')
        #ax.set_xlim(-0.50, 0.50)
        #ax.set_ylim(0, 0.30)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title(fr"Histogram of $Y^h$ for $\rho={rho}$")
        ax.set_xlabel(r"$Y^h$")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            fig.savefig(f"{save_dir}/hist_Yh_rho{rho}.pdf", format='pdf', bbox_inches='tight')
        plt.show()
        plt.close(fig)


def plot_hist_S_vs_rho(params, rho_list, num_sim=50000, seed= 42, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    """
    Plot histogram of S/N for each rho in rho_list.
    The y-axis shows relative frequency (density = count / num_sim).
    Optionally save each figure as 'hist_Yh_rho{rho}.pdf' in save_dir.
    """
    for rho in rho_list:
        N = params["N"]
        p = params.copy(); p["rho"] = rho
        per_policy_S = simulate_decomp(p, n_sim=50000, seed= seed)['S']/N

        fig, ax = plt.subplots(figsize=(6, 5))
        weights = np.ones_like(per_policy_S) / num_sim
        ax.hist(per_policy_S, bins=50, weights=weights, edgecolor='black')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 0.70)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title(fr"Histogram of per-policy liability for $\rho={rho}$")
        ax.set_xlabel(r"$\frac{S}{N}$")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            fig.savefig(f"{save_dir}/hist_S_rho{rho}.pdf", format='pdf', bbox_inches='tight')
        plt.show()
        plt.close(fig)


# In[12]:


rho_list = [-0.8, -0.4, 0, 0.4, 0.8]
plot_hist_Yh_vs_rho(params, rho_list, num_sim=50000, seed= 42, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")


# In[18]:


def plot_hist_ES_given_Y_vs_rho(params, rho_list, num_sim=50000, seed= 42, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    """
    Plot histogram of Y_h (residual) for each rho in rho_list.
    The y-axis shows relative frequency (density = count / num_sim).
    Optionally save each figure as 'hist_Yh_rho{rho}.pdf' in save_dir.
    """
    for rho in rho_list:
        N = params["N"]
        p = params.copy(); p["rho"] = rho
        decomp = simulate_decomp(p, n_sim=50000, seed= seed)
        ES_given_Y = decomp['Y_h'] + decomp['Y_s_fin']



        fig, ax = plt.subplots(figsize=(6, 5))
        weights = np.ones_like(ES_given_Y) / num_sim
        ax.hist(ES_given_Y, bins=50, weights=weights, edgecolor='black')
        #ax.set_xlim(-0.50, 0.50)
        #ax.set_ylim(0, 0.30)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title(fr"Histogram of $E[S|Y^{(1)}]$ for $\rho={rho}$")
        ax.set_xlabel(r"$E[S|Y^{(1)}]$")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            fig.savefig(f"{save_dir}/hist_ES_given_Y_rho{rho}.pdf", format='pdf', bbox_inches='tight')
        plt.show()
        plt.close(fig)

rho_list = [-0.8, -0.4, 0, 0.4, 0.8]
plot_hist_ES_given_Y_vs_rho(params, rho_list, num_sim=50000, seed= 42, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")


# In[6]:


def plot_hist_Ys_fin_vs_rho(params, rho_list, num_sim=50000, seed= 420, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    """
    Plot histogram of Y_s_fin for each rho in rho_list.
    The y-axis shows relative frequency (density = count / num_sim).
    Optionally save each figure as 'hist_Ys_fin_rho{rho}.pdf' in save_dir.
    """
    skews_Ys_fin_vs_rho = []
    kurtosis_Ys_fin_vs_rho = []
    variance_Ys_fin_vs_rho = []
    for rho in rho_list:
        N = params["N"]
        p = params.copy(); p["rho"] = rho
        Y_s_fin = simulate_decomp(p, n_sim=50000, seed= seed)['Y_s_fin']

        skews_Ys_fin_vs_rho.append(st.skew(Y_s_fin))
        kurtosis_Ys_fin_vs_rho.append(st.kurtosis(Y_s_fin, fisher=True))
        variance_Ys_fin_vs_rho.append(np.var(Y_s_fin, ddof = 1))

        fig, ax = plt.subplots(figsize=(6, 5))
        weights = np.ones_like(Y_s_fin) / num_sim
        ax.hist(Y_s_fin, bins=50, weights=weights, edgecolor='black')
        ax.set_xlim(-100, 200)
        ax.set_ylim(0, 0.30)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        #ax.set_title(fr"Histogram of $Y^s_{{fin}}$ for $\rho={rho}$")
        ax.set_xlabel(r"$Y^s_{fin}$")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            fig.savefig(f"{save_dir}/hist_Ys_fin_rho{rho}.pdf", format='pdf', bbox_inches='tight')
        plt.show()
        plt.close(fig)
    print(skews_Ys_fin_vs_rho, kurtosis_Ys_fin_vs_rho, variance_Ys_fin_vs_rho)



rho_list = [-0.8, 0, 0.8]
params = {
    "N":         500,
    "mu":        0.03,
    "sigma":     0.25,
    "T":         10.0,
    "mu_s":      mu_s,
    "sigma_s":   sigma_s,
    "rho":       0,
    "K":         1,
    "alpha":     0.5,
    "y0":        0.7,
    "r":         0.02
}
plot_hist_Ys_fin_vs_rho(params, rho_list, num_sim=50000, seed= 42, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")


# In[14]:


def plot_hist_Ys_act_vs_rho(params, rho_list, num_sim=50000, seed= 42, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    """
    Plot histogram of Y_s_res for each rho in rho_list.
    The y-axis shows relative frequency (density = count / num_sim).
    Optionally save each figure as 'hist_Ys_res_rho{rho}.pdf' in save_dir.
    """
    skews_Ys_act_vs_rho = []
    kurtosis_Ys_act_vs_rho = []
    variance_Ys_act_vs_rho = []
    for rho in rho_list:
        N = params["N"]
        p = params.copy(); p["rho"] = rho
        Y_s_act = simulate_decomp(p, n_sim=50000, seed= seed)['Y_s_act']
        skews_Ys_act_vs_rho.append(st.skew(Y_s_act))
        kurtosis_Ys_act_vs_rho.append(st.kurtosis(Y_s_act, fisher=True))
        variance_Ys_act_vs_rho.append(np.var(Y_s_act, ddof =1))

        fig, ax = plt.subplots(figsize=(6, 5))
        weights = np.ones_like(Y_s_act) / num_sim
        ax.hist(Y_s_act, bins=50, weights=weights, edgecolor='black')
        ax.set_xlim(-300, 300)
        ax.set_ylim(0, 0.25)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        #ax.set_title(fr"Histogram of $Y^s_{{res}}$ for $\rho={rho}$")
        ax.set_xlabel(r"$Y^s_{act}$")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            fig.savefig(f"{save_dir}/hist_Ys_res_rho{rho}.pdf", format='pdf', bbox_inches='tight')
        plt.show()
        plt.close(fig)
    print(skews_Ys_act_vs_rho, kurtosis_Ys_act_vs_rho, variance_Ys_act_vs_rho)



rho_list = [-0.8, 0, 0.8]
plot_hist_Ys_act_vs_rho(params, rho_list, num_sim=50000, seed= 42, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")


# In[11]:


# MVSD valuation in Example 1
def mvsd_valuation(params, beta, n_sim=200000, seed= 42):
    # Unpack parameters
    N       = params["N"]
    mu      = params["mu"]
    sigma   = params["sigma"]
    T       = params["T"]
    mu_s    = params["mu_s"]
    sigma_s = params["sigma_s"]
    rho     = params["rho"]
    K       = params["K"]
    alpha   = params["alpha"]
    y0      = params.get("y0", 1.0)
    r       = params["r"]

    # 1) draw underlying normals
    Y = np.random.randn(n_sim)
    m_y = mu_s + rho * sigma_s * Y
    s_y = sigma_s * np.sqrt(1 - rho**2)
    bar_Z = m_y + s_y * np.random.randn(n_sim)
    Z = np.where(bar_Z < 0, bar_Z, 0.0)

    # 2) terminal stock marginal under P
    Y1 = y0 * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Y)

    # 3) Bernoulli draws
    p = np.clip(np.exp(Z), 0.0, 1.0)
    X = np.random.binomial(1, p[:, None], size=(n_sim, N))
    Xsum = X.sum(axis=1)

    # 4) total payoff
    payoff_factor = 1 + alpha * np.maximum(Y1 - K, 0.0)
    S = payoff_factor * Xsum

    # --------------------------------------------------------------------
    # 5) closed-form theta's
    EY1   = E_Y1(params)
    VarY1 = Var_Y1(params)
    ES    = E_S(params)
    ESY1  = E_SY1(params)

    CovSY1 = ESY1 - ES * EY1
    theta1 = CovSY1 / VarY1
    theta0 = (ES - theta1 * EY1)/(np.exp(r * T))
    # --------------------------------------------------------------------
    # MVSD valuation
    hedge_value_per_policy = (theta1 * y0 + theta0 * 1) / N
    non_hedgeable_part = S - theta1 * Y1 - theta0 * np.exp(r * T)
    non_hedgeable_value_per_policy = beta * np.exp(-r * T) * np.sqrt(np.var(non_hedgeable_part)) / N
    mvsd_value_per_policy = hedge_value_per_policy + non_hedgeable_value_per_policy
    return {
        "mvsd_value": mvsd_value_per_policy,
        "mvsd_hedge": hedge_value_per_policy,
        "mvsd_residual": non_hedgeable_value_per_policy,
           }


def mvsd_valuation_vs_rho(params, rho_lists, beta, n_sim=200000, seed= 42):
    mvsd_value_vs_rho = []
    mvsd_hedge_vs_rho = []
    mvsd_residual_vs_rho = []
    for rho in rho_lists:
        p = params.copy()
        p["rho"] = rho
        mvsd_results = mvsd_valuation(p, beta, n_sim=200000, seed= 42)
        mvsd_value = mvsd_results["mvsd_value"]
        mvsd_hedge = mvsd_results["mvsd_hedge"]
        mvsd_residual = mvsd_results["mvsd_residual"]
        mvsd_value_vs_rho.append(mvsd_value)
        mvsd_hedge_vs_rho.append(mvsd_hedge)
        mvsd_residual_vs_rho.append(mvsd_residual)
    return {
        "mvsd_value_vs_rho": mvsd_value_vs_rho,
        "mvsd_hedge_vs_rho": mvsd_hedge_vs_rho,
        "mvsd_residual_vs_rho": mvsd_residual_vs_rho,
           }

def plot_mvsd_vs_rho(params, beta, n_sim = 200000, seed = 42,
                         save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    os.makedirs(save_dir, exist_ok=True)
    rhos = np.linspace(-0.99, 0.99, 10)  
    mvsd_vs_rho_results = mvsd_valuation_vs_rho(params, rhos, beta, n_sim, seed)

    plt.figure(figsize=(8,5))
    plt.plot(rhos, mvsd_vs_rho_results["mvsd_value_vs_rho"], label="MVSD valuation", lw=2)
    plt.plot(rhos, mvsd_vs_rho_results["mvsd_hedge_vs_rho"], label="MVSD hedge", lw=2)
    plt.xlabel("Correlation $\\rho$")
    plt.ylabel("Valuation")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/mvsd_valuation_vs_rho.pdf", bbox_inches="tight")
    plt.show()

plot_mvsd_vs_rho(params, beta = 0.06, n_sim = 200000, seed = 42,
                         save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")


# In[13]:


# TSSD valuation in Example 2
def tssd_valuation(params, beta, n_sim=200000, seed= 42):
    # Unpack parameters
    N       = params["N"]
    mu      = params["mu"]
    sigma   = params["sigma"]
    T       = params["T"]
    mu_s    = params["mu_s"]
    sigma_s = params["sigma_s"]
    rho     = params["rho"]
    K       = params["K"]
    alpha   = params["alpha"]
    y0      = params.get("y0", 1.0)
    r       = params["r"]

    # 1) draw underlying normals
    Y = np.random.randn(n_sim)
    m_y = mu_s + rho * sigma_s * Y
    s_y = sigma_s * np.sqrt(1 - rho**2)

    # 2) terminal stock marginal under P
    Y1 = y0 * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Y)

    # 4) total payoff
    payoff_factor = 1 + alpha * np.maximum(Y1 - K, 0.0)

    # 5) conditional expectation
    ES_given_Y1 = N * payoff_factor * (norm.cdf(m_y / s_y) + np.exp(m_y + 0.5 * s_y**2) * norm.cdf(-m_y / s_y - s_y))
    VarS_given_Y1 = payoff_factor**2 * ((N**2 - N) * (norm.cdf(m_y / s_y) + np.exp(2 * m_y + 2 * s_y**2) * norm.cdf(-m_y / s_y - 2 * s_y)) + 
                                        N * (norm.cdf(m_y / s_y) + np.exp(m_y + 0.5 * s_y**2) * norm.cdf(-m_y / s_y - s_y)) -
                                        N**2 * ((norm.cdf(m_y / s_y) + np.exp(m_y + 0.5 * s_y**2) * norm.cdf(-m_y / s_y - s_y))**2))
    VarS_given_Y1 = np.clip(VarS_given_Y1, a_min=0.0, a_max=None)

    # 6) Esscher transform to the risk-neutral measure Q
    veta = (r - mu) / (sigma**2)
    phi_f = np.exp(veta * (sigma*np.sqrt(T) * Y + (mu - 0.5*sigma**2)*T)) / np.exp(veta * (mu - 0.5*sigma**2)*T + 0.5 * veta**2 * sigma**2*T)

    # 7) risk-neutral expectation of Y_h + Y_s_fin
    E_Q_ES_given_Y1 = np.mean(ES_given_Y1 * phi_f)

    # 8) risk-neutral expectation of Var[Y_i + Y_s_res| Y]
    E_Q_sqrt_VarS_given_Y1 = np.mean(np.sqrt(VarS_given_Y1) * phi_f)

    # 9) TSSD valuation
    tssd_valuation_per_policy = np.exp(-r * T) / N * E_Q_ES_given_Y1 + np.exp(-r * T) / N * beta * E_Q_sqrt_VarS_given_Y1

    return tssd_valuation_per_policy


print(tssd_valuation(params, beta=0.06, n_sim=200000, seed= 42))

def tssd_valuation_vs_rho(params, rho_lists, beta, n_sim=200000, seed= 42):
    tssd_value_vs_rho = []
    for rho in rho_lists:
        p = params.copy()
        p["rho"] = rho
        tssd_value = tssd_valuation(p, beta, n_sim=200000, seed= 42)
        tssd_value_vs_rho.append(tssd_value)
    return tssd_value_vs_rho

rho_lists = [-0.8, 0, 0.8]
print(tssd_valuation_vs_rho(params, rho_lists, beta = 0.06, n_sim=100000, seed= 42))

def best_estimate_vs_rho(params, rho_lists):
    N = params["N"]
    r = params["r"]
    T = params["T"]
    best_estimate = []
    for rho in rho_lists:
        p = params.copy()
        p["rho"] = rho
        ES = E_S(p)
        ES_perpolicy_discounted = np.exp(-r*T)*ES/N
        best_estimate.append(ES_perpolicy_discounted)
    return best_estimate


# In[22]:


def plot_valuation_vs_rho(params, sigma_max, beta, n_sim = 200000, seed = 42,
                         save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    os.makedirs(save_dir, exist_ok=True)
    rhos = np.linspace(-0.99, 0.99, 10)

    N = params["N"]
    r = params["r"]

    mvsd_vs_rho_results = mvsd_valuation_vs_rho(params, rhos, beta, n_sim, seed)
    tssd_vs_rho_results = tssd_valuation_vs_rho(params, rhos, beta, n_sim, seed)
    newhb_vs_rho_results = newhb_valuation_vs_rho(params, sigma_max, rhos, beta=0.03, n_sim=200000, seed=42)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(rhos, mvsd_vs_rho_results["mvsd_hedge_vs_rho"], label="MV hedge", lw=2)
    ax.plot(rhos, mvsd_vs_rho_results["mvsd_value_vs_rho"], label="MVSD", lw=2)
    ax.plot(rhos, tssd_vs_rho_results, label="TSSD", lw=2)
    ax.plot(rhos, newhb_vs_rho_results, label="CMC", lw=2)
    ax.set_xlim(-1,1)
    ax.set_ylim(0.66, 0.72)
    ax.set_xlabel("Correlation $\\rho$")
    ax.set_ylabel("Valuation")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/valuation_vs_rho.pdf", bbox_inches="tight")
    plt.show()

def plot_relative_valuation_vs_rho(
    params,
    sigma_max: float,
    beta: float,
    n_sim: int = 200000,
    seed: int = 42,
    save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    """
    Plot MVSD, TSSD and NewMC valuations *relative* to the MV‐hedge benchmark = 100%,
    as functions of correlation rho, using a square background and 1:1 aspect ratio.
    """
    os.makedirs(save_dir, exist_ok=True)
    rhos = np.linspace(-0.99, 0.99, 10)

    # 1) compute all series
    mvsd = mvsd_valuation_vs_rho(params, rhos, beta, n_sim, seed)
    tssd = tssd_valuation_vs_rho(params, rhos, beta, n_sim, seed)
    newmc = newhb_valuation_vs_rho(params, sigma_max, rhos, beta, n_sim, seed)

    # 2) extract the MV-hedge benchmark and normalize
    mv_benchmark = np.array(mvsd["mvsd_hedge_vs_rho"])
    # percentages relative to benchmark
    mv_hedge_pct = mv_benchmark / mv_benchmark * 100.0      # all ones => 100%
    mvsd_pct     = np.array(mvsd["mvsd_value_vs_rho"]) / mv_benchmark * 100.0
    tssd_pct     = np.array(tssd) / mv_benchmark * 100.0
    newmc_pct    = np.array(newmc) / mv_benchmark * 100.0

    # 3) set style and figure
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8,5))

    # 4) plot
    ax.plot(rhos, mv_hedge_pct, label="MV Hedge (100 %)", lw=2, color="C0")
    ax.plot(rhos, mvsd_pct,     label="MVSD",               lw=2, color="C1")
    ax.plot(rhos, tssd_pct,     label="TSSD",               lw=2, color="C2")
    ax.plot(rhos, newmc_pct,    label="CMC",              lw=2, color="C3")

    # 5) labels & legend
    ax.set_xlim(-1,1)
    ax.set_xlabel("Correlation $\\rho$")
    ax.set_ylabel("Valuation (% of MV Hedge)")
    ax.set_ylim(98, 104)
    ax.legend(loc="upper left", frameon=True)

    # 7) save & show
    plt.tight_layout()
    fig.savefig(f"{save_dir}/relative_valuation_vs_rho.pdf", bbox_inches="tight")
    plt.show()

plot_valuation_vs_rho(params, sigma_max = 0.2625, beta = 0.03, n_sim = 200000, seed = 42,
                         save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")

plot_relative_valuation_vs_rho(params, sigma_max = 0.2625, beta = 0.03, n_sim = 200000, seed = 42,
                         save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")


# In[17]:


def E_Q_ES_given_Y(params,  sigma_Q):
    # We give the closed-form expression for E_Q_ES_given_Y
    # unpack parameters
    N       = params["N"]
    mu      = params["mu"]
    sigma   = params["sigma"]
    T       = params["T"]
    mu_s    = params["mu_s"]
    sigma_s = params["sigma_s"]
    rho     = params["rho"]
    K       = params["K"]
    alpha   = params["alpha"]
    y0      = params["y0"]
    r       = params["r"]

    # get the mean and variance of logY1 under the physical measure P
    mP = np.log(y0) + (mu - 0.5*sigma**2)*T
    vP = sigma**2 *T

    b = rho/(np.sqrt(vP) * np.sqrt(1-rho**2))
    s = sigma_s *  np.sqrt(1 - rho**2)
    a1 = mu_s/s - b * mP / np.sqrt(vP)
    a2 = -a1 - s

    # get the mean and variance of logY1 under a risk-neutral measure Q
    mQ = np.log(y0) + (r - 0.5*sigma_Q**2)*T
    vQ = sigma_Q**2 * T

    u1 = (a1 + b * mQ)/np.sqrt(1 + b**2 * vQ)
    v1 = (np.log(K) - mQ)/np.sqrt(vQ)
    u2 = u1 + b*vQ/np.sqrt(1 + b**2 * vQ)
    v2 = -v1 + np.sqrt(vQ)
    corr = b * np.sqrt(vQ)/np.sqrt(1 + b**2 * vQ)

    term1 = biv_cdf(u1, v1, -corr) + (1 - alpha*K) * biv_cdf(u1, -v1, corr) 
    term2 = alpha * np.exp(vQ/2 + mQ) * biv_cdf(u2, v2, corr)

    u3 = (a2 - b**2*s*vQ - b*mQ)/np.sqrt(1 + b**2 * vQ)
    v3 = (np.log(K) - b*s*vQ - mQ)/np.sqrt(vQ)
    u4 = u3 - b*vQ/np.sqrt(1 + b**2 * vQ)
    v4 = -v3 + np.sqrt(vQ)

    term3 = np.exp(b*s*mQ + (b*s)**2 * 0.5 * vQ) * (biv_cdf(u3, v3, corr) + (1 - alpha*K) * biv_cdf(u3, -v3, -corr))
    term4 = alpha * np.exp((b*s+1)*mQ + (b*s+1)**2 * 0.5 * vQ) * biv_cdf(u4, v4, -corr)

    E_Q_ES_given_Y = N * (term1 + term2) + N * np.exp(a1 * s + s**2 * 0.5) * (term3 + term4)
    return E_Q_ES_given_Y


def find_sup_E_Q_ES_given_Y(params,
                    sigmaQ_bounds=(1e-6, 1.0)):
    """
    Returns (sigmaQ_star, sup_val) where
      sup_val = max_{sigma_Q in sigmaQ_bounds} E_Q_ES_given_Y(params, sigma_Q)
      sigmaQ_star = argmax sigma_Q
    """
    # objective = negative of the function, since we maximize
    def obj(sigma_Q):
        return -E_Q_ES_given_Y(params, sigma_Q)

    res = minimize_scalar(obj,
                          bounds=sigmaQ_bounds,
                          method='bounded',
                          options={'xatol':1e-4})
    sigmaQ_star = res.x
    sup_val     = -res.fun
    return sigmaQ_star, sup_val

# Example usage:
mu_1 = 0.113826
sigma_1 = 0.002990
lambda_0 = 0.015030
T = 10
mu_s = -lambda_0 / mu_1 * (np.exp(mu_1 * T) - 1)
sigma_s = (sigma_1**2 / (2 * mu_1**3) * (np.exp(mu_1* 2 * T) - 4 * np.exp(mu_1 * T) + 3 + 2 * mu_1 * T))**0.5

params = {
    "N":         500,
    "mu":        0.03,
    "sigma":     0.25,
    "T":         10.0,
    "mu_s":      mu_s,
    "sigma_s":   sigma_s,
    "rho":       0,
    "K":         1,
    "alpha":     0.5,
    "y0":        0.7,
    "r":         0.02
}

sigmaQ_star, sup_E_Q_ES_given_Y = find_sup_E_Q_ES_given_Y(params,
                                       sigmaQ_bounds=(1e-3, 12)
                                                         )
print(f"sup E_Q[ES] = {sup_E_Q_ES_given_Y:.4f}  at sigma_Q = {sigmaQ_star:.4f}")


def E_Q_sigmaQ_independent(params):
    # Unpack parameters
    N       = params["N"]
    T       = params["T"]
    mu_s    = params["mu_s"]
    sigma_s = params["sigma_s"]
    K       = params["K"]
    alpha   = params["alpha"]
    y0      = params["y0"]
    r       = params["r"]
    c = mu_s/sigma_s
    sigma_Q_lists = np.linspace(0.0001, 6, 5000)
    E_Q_sigmaQ = []
    for sigma_Q in sigma_Q_lists:
        mQ = np.log(y0) + (r - 0.5*sigma_Q**2)*T
        vQ = sigma_Q**2 * T
        phi_1 = (-np.log(K) + mQ)/np.sqrt(vQ)
        phi_2 = phi_1 + np.sqrt(vQ)
        E_Q = N * (np.exp(mu_s + 0.5* sigma_s**2) * norm.cdf(-c-sigma_s) + norm.cdf(c)) * (1 + alpha*(y0 * np.exp(r*T) * norm.cdf(phi_2) - K *  norm.cdf(phi_1)))
        E_Q_sigmaQ.append(E_Q)
    return E_Q_sigmaQ


def plot_E_Q_sigmaQ_vs_rho(params, rho_list, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics"):
    os.makedirs(save_dir, exist_ok=True)
    sigma_Q_lists = np.linspace(0.0001, 6, 5000)
    E_Q_sigmaQ = E_Q_sigmaQ_independent(params)
    plt.figure(figsize=(8,5))
    plt.plot(sigma_Q_lists, E_Q_sigmaQ, label=fr"$\rho=0$", lw=1.5)
    for rho in rho_list:
        E_Q_sigmaQ = []
        p = params.copy()
        p["rho"] = rho
        for sigma_Q in sigma_Q_lists:
            E_Q = E_Q_ES_given_Y(p,  sigma_Q)
            E_Q_sigmaQ.append(E_Q)
        plt.plot(sigma_Q_lists, E_Q_sigmaQ, label=fr"$\rho={rho:.2f}$", lw=1.5)
    plt.axvline(x=0.25, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Risk-neutral volatility $\sigma_{Q}$")
    plt.ylabel("$E^Q[Y^h + Y^s_{fin}]$")
    plt.xlim(0,6)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/value_sigmaQ_vs_rho.pdf", bbox_inches="tight")
    plt.show()

rho_list = [-0.8, -0.4, 0.4, 0.8]
plot_E_Q_sigmaQ_vs_rho(params,rho_list, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")



# In[15]:


# New hedge_based valuation in Example 3
def newhb_valuation(params, sigma_max, beta=0.03, n_sim=200000, seed= 42):
    # Unpack parameters
    N       = params["N"]
    mu      = params["mu"]
    sigma   = params["sigma"]
    T       = params["T"]
    mu_s    = params["mu_s"]
    sigma_s = params["sigma_s"]
    rho     = params["rho"]
    K       = params["K"]
    alpha   = params["alpha"]
    y0      = params.get("y0", 1.0)
    r       = params["r"]

    # 1) draw underlying normals
    Y = np.random.randn(n_sim)
    m_y = mu_s + rho * sigma_s * Y
    s_y = sigma_s * np.sqrt(1 - rho**2)
    bar_Z = m_y + s_y * np.random.randn(n_sim)
    Z = np.where(bar_Z < 0, bar_Z, 0.0)

    # 2) terminal stock marginal under P
    Y1 = y0 * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Y)

    # 3) Bernoulli draws
    p = np.clip(np.exp(Z), 0.0, 1.0)
    X = np.random.binomial(1, p[:, None], size=(n_sim, N))
    Xsum = X.sum(axis=1)

    # 4) total payoff
    payoff_factor = 1 + alpha * np.maximum(Y1 - K, 0.0)
    S = payoff_factor * Xsum

    # --------------------------------------------------------------------
     # 5) analytic E[X1 | Y]
    I1 = norm.cdf(m_y / s_y)
    I2 = np.exp(m_y + 0.5*s_y**2) * norm.cdf(-m_y/s_y - s_y)
    EX1_given_Y = I1 + I2

    # 7) decompose
    fin_part = N * payoff_factor * EX1_given_Y

    # --------------------------------------------------------------------
    # ew hedge-based valuation
    fin_part_value = find_sup_E_Q_ES_given_Y(params,sigmaQ_bounds=(1e-6, sigma_max))[1]
    non_financial_value = np.var(S - fin_part)
    newhb_valuation_per_policy = np.exp(-r * T)* fin_part_value/N + beta * np.exp(-r * T) * np.sqrt(non_financial_value)/N
    return newhb_valuation_per_policy


def newhb_valuation_vs_rho(params, sigma_max, rho_lists, beta=0.03, n_sim=200000, seed= 42):
    newhb_value_vs_rho = []
    for rho in rho_lists:
        p = params.copy()
        p["rho"] = rho
        newhb_value = newhb_valuation(p,sigma_max, beta, n_sim=200000, seed= 42)
        newhb_value_vs_rho.append(newhb_value)
    return newhb_value_vs_rho


# In[16]:


def fin_part(params, Y1):
    # unpack parameters
    N       = params["N"]
    mu      = params["mu"]
    sigma   = params["sigma"]
    T       = params["T"]
    mu_s    = params["mu_s"]
    sigma_s = params["sigma_s"]
    rho     = params["rho"]
    K       = params["K"]
    alpha   = params["alpha"]
    y0      = params["y0"]
    r       = params["r"]

    m = mu_s + rho * sigma_s * (np.log(Y1) - np.log(y0) - (mu - 0.5* sigma**2) * T)/(sigma * np.sqrt(T))
    s = sigma_s * np.sqrt(1 - rho**2)
    fin_value = N * (1 + alpha * np.maximum(Y1 - K, 0.0))*(norm.cdf(m/s) + np.exp(m + 0.5*s**2)* norm.cdf(-m/s -s))
    return fin_value

def plot_Ys_fin_vs_Y1_rho(params, rho_list, save_dir="pics"):
    os.makedirs(save_dir, exist_ok=True)
    r, T = params["r"], params["T"]

    Y1_vals = np.linspace(0.01, 4.35, 5000)

    # 1) one figure for all lines
    plt.figure(figsize=(8,5))

    for rho in rho_list:
        # update params
        p = params.copy()
        p["rho"] = rho

        # compute hedge‐line
        EY1    = E_Y1(p)
        VarY1  = Var_Y1(p)
        ES     = E_S(p)
        ESY1   = E_SY1(p)
        CovSY1 = ESY1 - ES * EY1
        theta1 = CovSY1 / VarY1
        theta0 = (ES - theta1 * EY1) * np.exp(-r * T)

        # evaluate systematic part
        fin_vals = fin_part(p, Y1_vals)
        Yh_vals = theta0 * np.exp(r * T) + theta1 * Y1_vals
        Ys_fin    = fin_vals - Yh_vals

        # plot on the same axes
        plt.plot(Y1_vals, Ys_fin, lw=1.5, label=fr"$\rho={rho}$")

    # decorate once
    plt.xlabel(r"$Y^{(1)}$")
    plt.ylabel("value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.xlim(0,4.35)

    # 3) save a single file
    fname = os.path.join(save_dir, "Ys_fin_vs_Y1_all_rho.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.show()
    plt.close()

params = {
    "N":         500,
    "mu":        0.03,
    "sigma":     0.25,
    "T":         10.0,
    "mu_s":      mu_s,
    "sigma_s":   sigma_s,
    "rho":       0.8,
    "K":         1,
    "alpha":     0.5,
    "y0":        0.7,
    "r":         0.02
}


rho_list = [-0.8, 0, 0.8]
plot_Ys_fin_vs_Y1_rho(params, rho_list, save_dir="/Users/lingbiwen/Desktop/Decomposition/Pics")



# In[ ]:




