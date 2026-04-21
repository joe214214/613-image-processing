"""
Reproduces Section V.A of Aharon, Elad & Bruckstein (2006):
"K-SVD: An Algorithm for Designing Overcomplete Dictionaries
 for Sparse Representation", IEEE Trans. Signal Process.

Synthetic dictionary recovery experiment:
  - True dictionary D0 in R^(20 x 50)
  - 1500 training signals, sparsity L=3
  - 80 K-SVD / MOD iterations
  - 20 Monte Carlo trials per noise level
  - Atom recovery threshold eps=0.01
"""
import json, numpy as np
from sklearn.linear_model import orthogonal_mp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Experiment parameters (match paper Section V.A) ──────────
N_DIM      = 20    # signal dimension n
N_ATOMS    = 50    # dictionary size K
N_SIGNALS  = 1500  # training signals N
SPARSITY   = 3     # non-zeros per signal L
N_ITER     = 80    # dictionary learning iterations
N_TRIALS   = 20    # Monte Carlo trials
EPS        = 0.01  # recovery threshold
SNR_LEVELS = [None, 30, 20, 10]   # None = noiseless
# ─────────────────────────────────────────────────────────────

def make_dictionary(rng):
    D = rng.standard_normal((N_DIM, N_ATOMS))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    return D

def make_signals(D, rng):
    X = np.zeros((N_ATOMS, N_SIGNALS))
    for i in range(N_SIGNALS):
        idx = rng.choice(N_ATOMS, SPARSITY, replace=False)
        X[idx, i] = rng.standard_normal(SPARSITY)
    return D @ X

def add_noise(Y, snr_db, rng):
    if snr_db is None:
        return Y
    p_sig   = np.mean(Y ** 2)
    p_noise = p_sig * 10 ** (-snr_db / 10)
    return Y + rng.standard_normal(Y.shape) * np.sqrt(p_noise)

def count_recovered(D_true, D_hat):
    recovered = 0
    for k in range(D_true.shape[1]):
        sims = np.abs(D_true[:, k] @ D_hat)   # inner products with all learned atoms
        if np.max(sims) >= 1 - EPS:
            recovered += 1
    return recovered

def init_dict(Y, rng):
    idx = rng.choice(Y.shape[1], N_ATOMS, replace=False)
    D   = Y[:, idx].copy()
    nrm = np.linalg.norm(D, axis=0, keepdims=True)
    nrm[nrm < 1e-10] = 1.0
    D  /= nrm
    return D

# ── K-SVD training ────────────────────────────────────────────
def ksvd_train(Y, rng):
    D = init_dict(Y, rng)
    for _ in range(N_ITER):
        X = orthogonal_mp(D, Y, n_nonzero_coefs=SPARSITY)  # (K, N)
        for k in range(N_ATOMS):
            used = np.where(X[k, :] != 0)[0]
            if len(used) == 0:
                continue
            Xt          = X.copy()
            Xt[k, used] = 0
            E           = Y[:, used] - D @ Xt[:, used]
            U, S, Vt    = np.linalg.svd(E, full_matrices=False)
            D[:, k]     = U[:, 0]
            X[k, used]  = S[0] * Vt[0, :]
    return D

# ── MOD training (Method of Optimal Directions) ───────────────
def mod_train(Y, rng):
    D = init_dict(Y, rng)
    for _ in range(N_ITER):
        X   = orthogonal_mp(D, Y, n_nonzero_coefs=SPARSITY)
        D   = Y @ np.linalg.pinv(X)
        nrm = np.linalg.norm(D, axis=0, keepdims=True)
        nrm[nrm < 1e-10] = 1.0
        D  /= nrm
    return D

# ── Main loop ─────────────────────────────────────────────────
results = {}
print(f"{'SNR':>10} | {'K-SVD (mean±std)':>18} | {'MOD (mean±std)':>16}")
print("-" * 52)

for snr in SNR_LEVELS:
    ksvd_counts, mod_counts = [], []
    for trial in range(N_TRIALS):
        base = trial * 100
        rng0 = np.random.default_rng(base)
        D0   = make_dictionary(rng0)
        Y0   = make_signals(D0, rng0)
        Yn   = add_noise(Y0, snr, rng0)

        D_k = ksvd_train(Yn, np.random.default_rng(base + 1))
        D_m = mod_train (Yn, np.random.default_rng(base + 2))

        ksvd_counts.append(count_recovered(D0, D_k))
        mod_counts .append(count_recovered(D0, D_m))

    label = "No noise" if snr is None else f"{snr} dB"
    km, ks = np.mean(ksvd_counts), np.std(ksvd_counts)
    mm, ms = np.mean(mod_counts),  np.std(mod_counts)
    results[str(snr)] = {
        "ksvd_mean": round(km, 2), "ksvd_std": round(ks, 2),
        "mod_mean":  round(mm, 2), "mod_std":  round(ms, 2),
    }
    print(f"{label:>10} | {km:>6.1f} ± {ks:.1f}          | {mm:>5.1f} ± {ms:.1f}")

with open("synthetic_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved synthetic_results.json")

# ── Bar chart ─────────────────────────────────────────────────
labels     = ["No noise", "SNR 30 dB", "SNR 20 dB", "SNR 10 dB"]
ksvd_means = [results[str(s)]["ksvd_mean"] for s in SNR_LEVELS]
mod_means  = [results[str(s)]["mod_mean"]  for s in SNR_LEVELS]
ksvd_stds  = [results[str(s)]["ksvd_std"]  for s in SNR_LEVELS]
mod_stds   = [results[str(s)]["mod_std"]   for s in SNR_LEVELS]

x, w = np.arange(4), 0.35
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w/2, ksvd_means, w, yerr=ksvd_stds, label='K-SVD',
       color='steelblue', capsize=4, alpha=0.9)
ax.bar(x + w/2, mod_means,  w, yerr=mod_stds,  label='MOD',
       color='coral',     capsize=4, alpha=0.9)
ax.axhline(N_ATOMS, color='k', linestyle='--', linewidth=0.8,
           label=f'Perfect recovery ({N_ATOMS})')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel('Noise Level')
ax.set_ylabel('Atoms Recovered (out of 50)')
ax.set_title('Synthetic Dictionary Recovery: K-SVD vs MOD\n'
             r'$n=20,\ K=50,\ N=1500,\ L=3,\ 80\ \mathrm{iter},\ 20\ \mathrm{trials}$')
ax.set_ylim(0, 56); ax.legend()
plt.tight_layout()
plt.savefig('synthetic_recovery.png', dpi=150)
print("Saved synthetic_recovery.png")
