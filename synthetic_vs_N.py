"""
Reproduces paper Figure 6: atom recovery rate vs. number of training signals N.
Aharon, Elad & Bruckstein (2006), Section V.A.

Settings match the main synthetic experiment except N is varied.
"""
import json, numpy as np
from sklearn.linear_model import orthogonal_mp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_DIM    = 20
N_ATOMS  = 50
SPARSITY = 3
N_ITER   = 80
N_TRIALS = 20
EPS      = 0.01
N_VALUES = [100, 300, 500, 1000, 1500, 2000, 3000]

def make_dictionary(rng):
    D = rng.standard_normal((N_DIM, N_ATOMS))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    return D

def make_signals(D, n_signals, rng):
    X = np.zeros((N_ATOMS, n_signals))
    for i in range(n_signals):
        idx = rng.choice(N_ATOMS, SPARSITY, replace=False)
        X[idx, i] = rng.standard_normal(SPARSITY)
    return D @ X

def count_recovered(D_true, D_hat):
    recovered = 0
    for k in range(D_true.shape[1]):
        if np.max(np.abs(D_true[:, k] @ D_hat)) >= 1 - EPS:
            recovered += 1
    return recovered

def init_dict(Y, rng):
    idx = rng.choice(Y.shape[1], N_ATOMS, replace=False)
    D   = Y[:, idx].copy()
    nrm = np.linalg.norm(D, axis=0, keepdims=True)
    nrm[nrm < 1e-10] = 1.0
    D  /= nrm
    return D

def ksvd_train(Y, rng):
    D = init_dict(Y, rng)
    for _ in range(N_ITER):
        X = orthogonal_mp(D, Y, n_nonzero_coefs=SPARSITY)
        for k in range(N_ATOMS):
            used = np.where(X[k, :] != 0)[0]
            if len(used) == 0:
                continue
            Xt = X.copy(); Xt[k, used] = 0
            E  = Y[:, used] - D @ Xt[:, used]
            U, S, Vt = np.linalg.svd(E, full_matrices=False)
            D[:, k]    = U[:, 0]
            X[k, used] = S[0] * Vt[0, :]
    return D

def mod_train(Y, rng):
    D = init_dict(Y, rng)
    for _ in range(N_ITER):
        X   = orthogonal_mp(D, Y, n_nonzero_coefs=SPARSITY)
        D   = Y @ np.linalg.pinv(X)
        nrm = np.linalg.norm(D, axis=0, keepdims=True)
        nrm[nrm < 1e-10] = 1.0
        D  /= nrm
    return D

results = {}
print(f"{'N':>6} | {'K-SVD':>8} | {'MOD':>8}")
print("-" * 30)

for N in N_VALUES:
    ksvd_counts, mod_counts = [], []
    for trial in range(N_TRIALS):
        base = trial * 100
        rng0 = np.random.default_rng(base)
        D0   = make_dictionary(rng0)
        Y0   = make_signals(D0, N, rng0)

        D_k = ksvd_train(Y0, np.random.default_rng(base + 1))
        D_m = mod_train (Y0, np.random.default_rng(base + 2))

        ksvd_counts.append(count_recovered(D0, D_k))
        mod_counts .append(count_recovered(D0, D_m))

    km, ks = np.mean(ksvd_counts), np.std(ksvd_counts)
    mm, ms = np.mean(mod_counts),  np.std(mod_counts)
    results[N] = {
        "ksvd_mean": round(km, 2), "ksvd_std": round(ks, 2),
        "mod_mean":  round(mm, 2), "mod_std":  round(ms, 2),
    }
    print(f"{N:>6} | {km:>5.1f}±{ks:.1f}  | {mm:>5.1f}±{ms:.1f}")

with open("recovery_vs_N.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
print("\nSaved recovery_vs_N.json")

ksvd_means = [results[N]["ksvd_mean"] for N in N_VALUES]
ksvd_stds  = [results[N]["ksvd_std"]  for N in N_VALUES]
mod_means  = [results[N]["mod_mean"]  for N in N_VALUES]
mod_stds   = [results[N]["mod_std"]   for N in N_VALUES]

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(N_VALUES, ksvd_means, yerr=ksvd_stds, marker='o',
            color='steelblue', capsize=4, label='K-SVD')
ax.errorbar(N_VALUES, mod_means,  yerr=mod_stds,  marker='s',
            color='coral',     capsize=4, label='MOD')
ax.axhline(N_ATOMS, color='k', linestyle='--', linewidth=0.8,
           label=f'Perfect recovery ({N_ATOMS})')
ax.set_xlabel('Number of Training Signals N')
ax.set_ylabel('Atoms Recovered (out of 50)')
ax.set_title('Recovery Rate vs. Training Set Size\n'
             r'$n=20,\ K=50,\ L=3,\ 80\ \mathrm{iter},\ 20\ \mathrm{trials},\ \mathrm{noiseless}$')
ax.set_ylim(0, 56); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('recovery_vs_N.png', dpi=150)
print("Saved recovery_vs_N.png")
