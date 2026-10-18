import json
import warnings

import numpy as np
from picard import picard
from sklearn.decomposition import PCA


class ComponentSpace:
    def __init__(
        self, dimensions=32, pca_dimensions=64, seed=0, solver="randomized", max_iter=3000
    ):
        if not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("Component count must be a positive integer")
        self.dimensions = dimensions
        self.pca_dimensions = max(dimensions, pca_dimensions)
        self.seed = seed
        self.solver = solver
        self.max_iter = max_iter
        self.score_scale = np.ones(dimensions)

    def fit(self, X, rows=None):
        X = self._matrix(X)
        selected = X if rows is None else X[rows]
        if len(selected) <= self.dimensions or X.shape[1] < self.dimensions:
            raise ValueError("Need more fitting examples than components")
        count = min(self.pca_dimensions, len(selected) - 1, X.shape[1])
        pca = PCA(n_components=count, svd_solver=self.solver, random_state=0, iterated_power=7)
        pca.fit(selected)
        self.mean = pca.mean_
        self.directions = pca.components_[: self.dimensions]
        self.variance = pca.explained_variance_[: self.dimensions]
        if np.any(self.variance <= 0):
            raise ValueError("The fitting data has too few varying directions")
        white = pca.transform(selected)[:, : self.dimensions] / np.sqrt(self.variance)

        # the same operation for text embeddings and image embeddings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, self.rotation, _ = picard(
                white.T,
                whiten=False,
                centering=False,
                ortho=True,
                extended=True,
                random_state=self.seed,
                max_iter=self.max_iter,
                tol=1e-7,
            )
        self.messages = [str(item.message) for item in caught]
        self.score_scale = np.ones(self.dimensions)
        return self

    def transform(self, X):
        X = self._matrix(X)
        if not hasattr(self, "rotation"):
            raise ValueError("Fit or load a component space first")
        if X.shape[1] != len(self.mean):
            raise ValueError("Embedding dimension does not match the fitted space")
        centered = X - self.mean
        projected = centered @ self.directions.T
        white = projected / np.sqrt(self.variance)
        return (white @ self.rotation.T) / self.score_scale

    def fit_transform(self, X, rows=None):
        self.fit(X, rows)
        return self.transform(X)

    def save(self, path):
        config = {
            "dimensions": self.dimensions,
            "pca_dimensions": self.pca_dimensions,
            "seed": self.seed,
            "solver": self.solver,
            "max_iter": self.max_iter,
        }
        np.savez(
            path,
            mean=self.mean,
            directions=self.directions,
            variance=self.variance,
            rotation=self.rotation,
            score_scale=self.score_scale,
            config=json.dumps(config),
            messages=json.dumps(self.messages),
        )

    @classmethod
    def load(cls, path, dimensions=32):
        with np.load(path, allow_pickle=False) as saved:
            if "config" in saved:
                space = cls(**json.loads(saved["config"].item()))
                space.mean = saved["mean"]
                space.directions = saved["directions"]
                space.variance = saved["variance"]
                space.rotation = saved["rotation"]
                space.score_scale = saved["score_scale"]
                space.messages = json.loads(saved["messages"].item())
            else:
                # accept the earlier novel result so its component numbering is preserved
                space = cls(dimensions=dimensions)
                space.mean = saved["pca_mean"]
                space.directions = saved["pca_components"][:dimensions]
                space.variance = saved["variance"][:dimensions]
                space.rotation = saved[f"rotation_{dimensions}"]
                space.messages = []
        return space

    @staticmethod
    def _matrix(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
            raise ValueError("Expected a nonempty matrix of finite embeddings")
        return X


def percentiles(scores):
    order = np.argsort(scores)
    out = np.empty(len(scores))
    out[order] = np.linspace(0, 1, len(scores))
    return out
