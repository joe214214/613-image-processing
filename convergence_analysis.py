"""
Convergence analysis: reconstruction error vs. iteration for K-SVD and MOD.
Reproduces the convergence behaviour discussed in Aharon et al. 2006.
Records ||Y - DX||_F^2 every 5 iterations over 80 total.
"""
import json, numpy as np
from sklearn.linear_model import orthogonal_mp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_DIM    = 20
N_ATOMS  = 50
N_SIGNALS= 1500
SPARSITY = 3
N_ITER   = 80
RECORD_AT= list(range(5, N_ITER + 1, 5))   # [5, 10, ..., 80]
N_TRIALS = 5   # fewer trials — we want the curve, not statistics
EPS      = 1e-10

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

def init_dict(Y, rng):
    idx = rng.choice(Y.shape[1], N_ATOMS, replace=False)
    D   = Y[:, idx].copy()
    nrm = np.linalg.norm(D, axis=0, keepdims=True)
    nrm[nrm < EPS] = 1.0
    return D / nrm

def reconstruction_error(Y, D, X):
    return float(np.linalg.norm(Y - D @ X, 'fro') ** 2) / Y.shape[1]

def ksvd_errors(Y, rng):
    D = init_dict(Y, rng)
    errors = []
    for it in range(1, N_ITER + 1):
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
        if it in RECORD_AT:
            errors.append(reconstruction_error(Y, D, X))
    return errors

def mod_errors(Y, rng):
    D = init_dict(Y, rng)
    errors = []
    for it in range(1, N_ITER + 1):
        X   = orthogonal_mp(D, Y, n_nonzero_coefs=SPARSITY)
        D   = Y @ np.linalg.pinv(X)
        nrm = np.linalg.norm(D, axis=0, keepdims=True)
        nrm[nrm < EPS] = 1.0
        D  /= nrm
        if it in RECORD_AT:
            errors.append(reconstruction_error(Y, D, X))
    return errors

ksvd_all, mod_all = [], []
for trial in range(N_TRIALS):
    rng0 = np.random.default_rng(trial * 100)
    D0   = make_dictionary(rng0)
    Y    = make_signals(D0, rng0)
    ksvd_all.append(ksvd_errors(Y, np.random.default_rng(trial * 100 + 1)))
    mod_all .append(mod_errors (Y, np.random.default_rng(trial * 100 + 2)))
    print(f"Trial {trial+1}/{N_TRIALS} done")

ksvd_mean = np.mean(ksvd_all, axis=0)
mod_mean  = np.mean(mod_all,  axis=0)

results = {"iterations": RECORD_AT,
           "ksvd_error": list(ksvd_mean),
           "mod_error":  list(mod_mean)}
with open("convergence_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved convergence_results.json")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(RECORD_AT, ksvd_mean, 'o-', color='steelblue', label='K-SVD')
ax.plot(RECORD_AT, mod_mean,  's-', color='coral',     label='MOD')
ax.set_xlabel('Iteration')
ax.set_ylabel(r'Mean Reconstruction Error $\|Y-DX\|_F^2 / N$')
ax.set_title('Convergence: K-SVD vs MOD\n'
             r'$n=20,\ K=50,\ N=1500,\ L=3$, noiseless, 5 trials')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence_curves.png', dpi=150)
print("Saved convergence_curves.png")
