import numpy as np


def omp(D, y, n_nonzero):
    residual = y.copy()
    selected = []
    for _ in range(n_nonzero):
        correlations = D.T @ residual
        idx = int(np.argmax(np.abs(correlations)))
        if idx in selected:
            break
        selected.append(idx)
        D_selected = D[:, selected]
        coefs, _, _, _ = np.linalg.lstsq(D_selected, y, rcond=None)
        residual = y - D_selected @ coefs
    x = np.zeros(D.shape[1])
    if selected:
        x[selected] = coefs
    return x


def ksvd(Y, n_atoms, n_nonzero, n_iter=10):
    signal_dim = Y.shape[0]

    # 初始化字典
    D = np.random.randn(signal_dim, n_atoms)
    D /= np.linalg.norm(D, axis=0, keepdims=True)

    for iteration in range(n_iter):
        print(f"  Iteration {iteration+1}/{n_iter}", flush=True)

        # Step 1: OMP 稀疏编码
        X = np.zeros((n_atoms, Y.shape[1]))
        for i in range(Y.shape[1]):
            X[:, i] = omp(D, Y[:, i], n_nonzero)

        # Step 2: 逐列更新字典
        for j in range(n_atoms):
            used = np.where(X[j, :] != 0)[0]
            if len(used) == 0:
                continue
            X_temp = X.copy()
            X_temp[j, used] = 0
            E = Y[:, used] - D @ X_temp[:, used]
            U, S, Vt = np.linalg.svd(E, full_matrices=False)
            D[:, j] = U[:, 0]
            X[j, used] = S[0] * Vt[0, :]

    return D, X
