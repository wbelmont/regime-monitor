"""Statistical Jump Model (JM) for market-regime identification.

Faithful implementation of the discrete jump model as specified in
Shu & Mulvey (Princeton), "Modeling Regime Changes in Financial Markets Using
Statistical Jump Models" (building on Nystrup et al. 2020; Bemporad et al.).

Objective (their eq. 2.1), with squared-Euclidean loss to K cluster centroids:

    minimize  sum_t || x_t - mu_{s_t} ||^2  +  lambda * sum_t 1[s_t != s_{t-1}]

We solve it with the paper's coordinate-descent scheme (their Algorithm 2):
  (a) fix the state sequence  -> recompute centroids as cluster means;
  (b) fix the centroids       -> solve for the optimal state sequence by the
      dynamic program in their Algorithm 3 / eq. (2.2):
          V(t, k) = l_{t,k} + min_j ( V(t-1, j) + lambda * 1[j != k] ).
Centroids are initialized with k-means++ and the fit is repeated from `n_init`
random restarts, keeping the lowest-objective solution (also per the paper).

All features are backward-looking, so `fit`/`predict` only ever use the data
passed in — enabling the leak-free *online inference* the paper requires (it
explicitly warns that forward-looking features inflate performance).
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class JumpModel:
    def __init__(
        self,
        n_states: int = 2,
        jump_penalty: float = 50.0,
        n_init: int = 10,
        max_iter: int = 50,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.centroids_: np.ndarray | None = None

    # ---- core dynamic program ------------------------------------------- #
    def _best_path(self, dist2: np.ndarray) -> np.ndarray:
        """Given per-day squared distances to each centroid (T x K), find the
        state sequence minimizing fit cost + jump penalties.

        Vectorized across states: for each step we build the (K_prev x K_curr)
        transition cost matrix in one shot (penalty off-diagonal, 0 on-diagonal)
        and take a column-wise argmin. Identical result to the naive double loop,
        but much faster — important because this runs on every walk-forward step.
        """
        T, K = dist2.shape
        # transition penalty matrix: 0 to stay, jump_penalty to switch.
        trans = self.jump_penalty * (1.0 - np.eye(K))
        cost = np.empty((T, K))
        back = np.zeros((T, K), dtype=int)
        cost[0] = dist2[0]
        for t in range(1, T):
            # total[j, k] = cost[t-1, j] + penalty(j->k)
            total = cost[t - 1][:, None] + trans
            j = np.argmin(total, axis=0)  # best previous state per current k
            back[t] = j
            cost[t] = dist2[t] + total[j, np.arange(K)]
        # backtrack
        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmin(cost[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        return path

    def _fit_once(
        self, X: np.ndarray, seed: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        from sklearn.cluster import kmeans_plusplus

        T, _ = X.shape
        # Initialize centroids with k-means++ (Arthur & Vassilvitskii, 2007),
        # exactly as Shu & Mulvey specify for the coordinate-descent JM fit.
        centroids, _ = kmeans_plusplus(X, n_clusters=self.n_states, random_state=seed)
        centroids = centroids.copy()
        path = np.zeros(T, dtype=int)
        for _ in range(self.max_iter):
            dist2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_path = self._best_path(dist2)
            # recompute centroids
            for k in range(self.n_states):
                members = X[new_path == k]
                if len(members):
                    centroids[k] = members.mean(axis=0)
            if np.array_equal(new_path, path):
                break
            path = new_path
        # total cost for model selection across inits
        dist2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        fit_cost = dist2[np.arange(T), path].sum()
        jumps = (np.diff(path) != 0).sum() * self.jump_penalty
        return centroids, path, float(fit_cost + jumps)

    def fit(self, X) -> "JumpModel":
        Xs = self.scaler.fit_transform(np.asarray(X, dtype=float))
        best = None
        # Run the coordinate descent from `n_init` k-means++ starts and keep the
        # solution with the best objective (Shu & Mulvey use 10 restarts).
        for r in range(self.n_init):
            centroids, path, total = self._fit_once(Xs, seed=self.random_state + r)
            if best is None or total < best[2]:
                best = (centroids, path, total)
        assert best is not None
        self.centroids_ = best[0]
        self._train_path = best[1]
        return self

    def predict(self, X) -> np.ndarray:
        """Label new data using fitted centroids (with the jump penalty applied
        across the provided sequence)."""
        assert self.centroids_ is not None, "Call fit() first."
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        dist2 = ((Xs[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self._best_path(dist2)


# ===================================================================== #
#  Continuous Jump Model (CJM) — Shu & Mulvey, Section 2.4 / Algorithm 4
# ===================================================================== #
class ContinuousJumpModel:
    r"""Continuous statistical jump model (Shu & Mulvey, eqs. 2.5–2.8).

    Generalizes the discrete JM by letting each period's hidden state be a
    *probability vector* on the simplex rather than a hard label. We solve

        argmin_{Theta, S}  sum_t  sum_k  s_{t,k} * l(y_t, theta_k)
                           + (lambda/4) * sum_t || s_{t-1} - s_t ||_1^2
        s.t.  S >= 0,  rows of S sum to 1

    by coordinate descent (their Algorithm 4):
      (a) fix S -> centroids are the probability-weighted means (eq. 2.7);
      (b) fix Theta -> solve the convex state-sequence problem (eq. 2.8).

    The reduction factor 1/4 keeps `lambda` on the same scale as the discrete
    model. Setting the probabilities to 0/1 recovers the discrete JM exactly.

    `predict_proba` returns the per-period regime probabilities — the quantity
    the paper highlights as most valuable for risk management, and exactly what
    this tool's bear-probability signal needs.
    """

    def __init__(
        self,
        n_states: int = 2,
        jump_penalty: float = 50.0,
        n_init: int = 10,
        max_iter: int = 30,
        state_max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.n_init = n_init
        self.max_iter = max_iter  # coordinate-descent iterations
        self.state_max_iter = state_max_iter  # inner solver iterations
        self.tol = tol
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.centroids_: np.ndarray | None = None

    # ---- loss matrix: scaled squared L2 distance to each centroid -------- #
    @staticmethod
    def _loss_matrix(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        # L[t, k] = 0.5 * || x_t - mu_k ||^2  (the 0.5 scaling matches the paper)
        return 0.5 * ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

    # ---- (b) state-sequence step: convex QP on the simplex (eq. 2.8) ----- #
    def _fit_states(self, L: np.ndarray, S0: np.ndarray | None = None) -> np.ndarray:
        """Minimize <S, L> + (lambda/4) * sum_t ||s_{t-1}-s_t||_1^2 over the
        simplex, by projected gradient descent (the objective is convex).

        `S0` warm-starts the solver (reused across coordinate-descent steps),
        which makes convergence fast on real data.
        """
        T, K = L.shape
        lam = self.jump_penalty
        S = _softmax(-L) if S0 is None else S0.copy()
        lr = 1.0 / (1.0 + lam)
        prev_obj = np.inf
        # relative tolerance scaled by the number of periods (objective grows
        # with T), so the early-stop triggers consistently regardless of window.
        rtol = self.tol * max(T, 1)
        for _ in range(self.state_max_iter):
            grad = L.copy()
            if T > 1:
                diff = np.diff(S, axis=0)  # s_t - s_{t-1}, shape (T-1, K)
                g = np.zeros_like(S)
                g[:-1] -= diff  # contribution to s_{t-1}
                g[1:] += diff  # contribution to s_t
                grad = grad + (lam / 2.0) * g
            S = _project_rows_to_simplex(S - lr * grad)
            obj = float((S * L).sum())
            if T > 1:
                d = np.diff(S, axis=0)
                obj += (lam / 4.0) * (np.abs(d).sum(axis=1) ** 2).sum()
            if abs(prev_obj - obj) < rtol:
                break
            prev_obj = obj
        return S

    def _fit_once(self, X: np.ndarray, seed: int):
        from sklearn.cluster import kmeans_plusplus

        T, _ = X.shape
        centroids, _ = kmeans_plusplus(X, n_clusters=self.n_states, random_state=seed)
        centroids = centroids.copy()
        S = None
        prev_obj = np.inf
        for _ in range(self.max_iter):
            # (b) state step — warm-start from the previous iteration's S.
            L = self._loss_matrix(X, centroids)
            S = self._fit_states(L, S0=S)
            # (a) parameter step: probability-weighted means (eq. 2.7)
            w = S.sum(axis=0)  # (K,)
            w_safe = np.where(w > 0, w, 1.0)
            centroids = (S.T @ X) / w_safe[:, None]
            # objective for convergence + model selection
            obj = float((S * L).sum())
            if T > 1:
                d = np.diff(S, axis=0)
                obj += (self.jump_penalty / 4.0) * (np.abs(d).sum(axis=1) ** 2).sum()
            if abs(prev_obj - obj) < self.tol * max(T, 1):
                break
            prev_obj = obj
        return centroids, S, prev_obj

    def fit(self, X) -> "ContinuousJumpModel":
        Xs = self.scaler.fit_transform(np.asarray(X, dtype=float))
        best = None
        for r in range(self.n_init):
            centroids, S, obj = self._fit_once(Xs, seed=self.random_state + r)
            if best is None or obj < best[2]:
                best = (centroids, S, obj)
        assert best is not None
        self.centroids_ = best[0]
        self._train_proba = best[1]
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return the (T x K) regime probability matrix for new data, applying
        the jump penalty across the provided sequence (online inference)."""
        assert self.centroids_ is not None, "Call fit() first."
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        L = self._loss_matrix(Xs, self.centroids_)
        return self._fit_states(L)

    def predict(self, X) -> np.ndarray:
        """Hard label = argmax of the probability vector (consistent with
        predict_proba, unlike an HMM's Viterbi-vs-forward-backward mismatch)."""
        return self.predict_proba(X).argmax(axis=1)


# ----------------------------- helpers ---------------------------------- #
def _softmax(A: np.ndarray) -> np.ndarray:
    A = A - A.max(axis=1, keepdims=True)
    E = np.exp(A)
    return E / E.sum(axis=1, keepdims=True)


def _project_rows_to_simplex(V: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row onto the probability simplex
    (Wang & Carreira-Perpinan, 2013)."""
    n, K = V.shape
    U = np.sort(V, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, K + 1)
    cond = U - cssv / ind > 0
    rho = cond.sum(axis=1)
    theta = cssv[np.arange(n), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0.0)
