from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor

# Helper for Pareto front (non-dominated sorting)
# Assumes minimization
def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Find the Pareto-efficient points.
    :param costs: An (n_points, n_objectives) array of costs.
                  Assumes minimization for all objectives.
    :return: A boolean array of shape (n_points,).
             True if the point is Pareto-efficient, False otherwise.
    """
    num_points = costs.shape[0]
    if num_points == 0:
        return np.array([], dtype=bool)

    is_efficient = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        # If point i is already marked as dominated, skip it
        if not is_efficient[i]:
            continue
        # Check if point i is dominated by any other point j
        for j in range(num_points):
            if i == j:
                continue # A point cannot dominate itself
            
            # Check if point j dominates point i
            # j dominates i if:
            #   all objectives of j are less than or equal to objectives of i (costs[j] <= costs[i])
            #   AND at least one objective of j is strictly less than objectives of i (costs[j] < costs[i])
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                is_efficient[i] = False
                break # Point i is dominated, no need to check further against other points
    return is_efficient

# Helper to scale points from [0,1]^dim to actual bounds
def _scale_points_to_bounds(points_unit_cube: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    scaled_points = lower_bounds + points_unit_cube * (upper_bounds - lower_bounds)
    return scaled_points

# Helper to scale points from actual bounds to [0,1]^dim
def _scale_points_to_unit_cube(points_scaled: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    # Add a small epsilon to denominator to prevent division by zero for zero-range dimensions
    range_bounds = upper_bounds - lower_bounds
    range_bounds[range_bounds == 0] = 1e-9 # Prevent division by zero
    unit_cube_points = (points_scaled - lower_bounds) / range_bounds
    return unit_cube_points


class MOBOEnsembleRidge_MPFDUWS:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None,
                 n_candidates_per_iteration: int = 500,
                 batch_size: int = 5,
                 ridge_window_size: int = 250,
                 acquisition_kappa: float = 2.0,
                 n_estimators_ridge: int = 30,
                 alpha_ridge: float = 1.0,
                 max_samples_ratio: float = 0.8,
                 lambda_param: float = 0.1): # New hyperparameter for weighted sum
        # Fixed problem parameters
        self.budget = budget
        self.dim = dim
        # bounds has shape (2, dim), bounds[0]: lower bound, bounds[1]: upper bound
        # The environment (evaluator / problem provider) should pass the true bounds.
        # Do NOT overwrite self.bounds with hard-coded values.
        if bounds is None:
            # Fallback: assume a simple [0, 1]^dim box if no bounds are provided.
            self.bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        else:
            self.bounds = np.asarray(bounds, dtype=float)

        # Hyperparameters (tuned by SMAC, defined in Space)
        self.n_candidates_per_iteration = n_candidates_per_iteration
        self.batch_size = batch_size
        self.ridge_window_size = ridge_window_size
        self.acquisition_kappa = acquisition_kappa
        self.n_estimators_ridge = n_estimators_ridge
        self.alpha_ridge = alpha_ridge
        self.max_samples_ratio = max_samples_ratio
        self.lambda_param = lambda_param # New hyperparameter

        # The number of objectives (self.n_obj) is unknown a priori.
        # It MUST be inferred on the first call to func inside _evaluate_points.
        self.n_obj: int | None = None

        # X has shape (n_points, n_dims), y has shape (n_points, n_obj)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.n_evals = 0  # the number of function evaluations

        # Choose a reasonable number of initial evaluations.
        # Use a small, budget-aware design (for example, proportional to dim, but not more than a fraction of the budget).
        self.n_init = min(max(2 * self.dim + 1, 5), self.budget // 4, 50)

        # BaggingRegressor models with Ridge base estimators for each objective
        self.bagging_ridges: list[BaggingRegressor] = []
        # Scaler for input features X
        self.scaler_X: StandardScaler | None = None

    def _sample_points(self, n_points: int) -> np.ndarray:
        # Sample n_points candidate points efficiently within self.bounds.
        # Use self.bounds[0] as lower bounds and self.bounds[1] as upper bounds.
        # Return array of shape (n_points, n_dims).
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=np.random.randint(0, 100000))
        points_unit_cube = sampler.random(n_points)
        scaled_points = _scale_points_to_bounds(points_unit_cube, self.bounds)
        return scaled_points

    def _predict_bagging_ridge_with_uncertainty(self, bagging_ridge_model: BaggingRegressor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation from a BaggingRegressor with Ridge base estimators.
        Uncertainty is estimated from the standard deviation of individual base estimator predictions.
        """
        # Get predictions from each base estimator
        # BaggingRegressor stores its fitted base estimators in .estimators_
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = np.array([estimator.predict(X) for estimator in bagging_ridge_model.estimators_])
        
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)
        
        # Add a small epsilon to sigma to prevent division by zero or overly optimistic bounds
        sigma = np.maximum(sigma, 1e-6)
        return mu, sigma

    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray):
        # Fit a surrogate model on (X, y).
        # Apply sliding window
        window_size = min(self.ridge_window_size, X_train.shape[0])
        X_window = X_train[-window_size:]
        y_window = y_train[-window_size:]

        # Scale input features X
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X_window)

        self.bagging_ridges = []
        for i in range(self.n_obj):
            # Base estimator is Ridge
            base_ridge = Ridge(alpha=self.alpha_ridge, random_state=42)
            
            bagging = BaggingRegressor(base_estimator=base_ridge,
                                       n_estimators=self.n_estimators_ridge,
                                       max_samples=self.max_samples_ratio, # Fraction of samples to draw
                                       random_state=42, # For reproducibility of bootstrapping
                                       n_jobs=-1) # Use all available cores for parallel fitting of estimators
            bagging.fit(X_scaled, y_window[:, i])
            self.bagging_ridges.append(bagging)

    def _acquisition_function(self, X_candidates: np.ndarray) -> np.ndarray:
        # Implement a multi-objective acquisition function (Maximin-Pareto-Front-Distance with Uncertainty and Weighted Sum).
        if not self.bagging_ridges or self.X is None or self.y is None or self.y.shape[0] == 0:
            # If no models or no data, return random scores for exploration
            return np.random.rand(X_candidates.shape[0])

        # Scale candidate inputs
        X_candidates_scaled = self.scaler_X.transform(X_candidates)

        # Predict mean and std for all objectives
        mu_preds = np.zeros((X_candidates.shape[0], self.n_obj))
        sigma_preds = np.zeros((X_candidates.shape[0], self.n_obj))
        for i, bagging_ridge_model in enumerate(self.bagging_ridges):
            mu, sigma = self._predict_bagging_ridge_with_uncertainty(bagging_ridge_model, X_candidates_scaled)
            mu_preds[:, i] = mu
            sigma_preds[:, i] = sigma

        # CRITICAL ROBUSTNESS: Normalize objectives dynamically
        # Ideal point (min observed) and Nadir point (max observed) from *all* observed data self.y
        ideal_point = np.min(self.y, axis=0)
        nadir_point = np.max(self.y, axis=0)
        range_y = nadir_point - ideal_point + 1e-9 # Add epsilon to avoid division by zero

        # Normalize predicted means and standard deviations
        mu_preds_norm = (mu_preds - ideal_point) / range_y
        sigma_preds_norm = sigma_preds / range_y

        # Get current non-dominated set for acquisition, based on the sliding window for efficiency
        y_for_pf_acquisition = self.y[-min(self.ridge_window_size, self.y.shape[0]):]
        pareto_indices_current = is_pareto_efficient(y_for_pf_acquisition)
        
        # Normalize the current non-dominated set using ideal/nadir from *all* data
        P_ND_y_norm = (y_for_pf_acquisition[pareto_indices_current] - ideal_point) / range_y
        
        if P_ND_y_norm.shape[0] == 0: # If no Pareto points in the window, return random scores for exploration
             return np.random.rand(X_candidates.shape[0])

        acquisition_values = np.zeros(X_candidates.shape[0])

        for i in range(X_candidates.shape[0]):
            # Calculate LCB for normalized objectives (minimization: mu - kappa * sigma)
            lcb_vector = mu_preds_norm[i, :] - self.acquisition_kappa * sigma_preds_norm[i, :]
            
            # Diversity Term: Maximin-Pareto-Front-Distance
            # We want to maximize the minimum distance to encourage finding new regions.
            distances = np.linalg.norm(P_ND_y_norm - lcb_vector, axis=1)
            diversity_score = np.min(distances)

            # Quality Term: Weighted sum of LCB objectives (smaller is better, so maximize negative sum)
            quality_score = -np.sum(lcb_vector)
            
            # Combine diversity and quality
            acquisition_values[i] = diversity_score + self.lambda_param * quality_score
        
        return acquisition_values

    def _select_next_points(self, batch_size: int) -> np.ndarray:
        # Select the next points to evaluate using the acquisition function.
        # Generate candidate points and score them with the acquisition function.
        # The selection strategy can be any heuristic / evolutionary / mathematical / hybrid method.
        # Return an array of shape (batch_size, n_dims).

        # Generate a large set of candidate points
        candidates_unit_cube = qmc.Sobol(d=self.dim, scramble=True, seed=np.random.randint(0, 100000)).random(self.n_candidates_per_iteration)
        X_candidates = _scale_points_to_bounds(candidates_unit_cube, self.bounds)

        # Calculate acquisition values
        acquisition_scores = self._acquisition_function(X_candidates)

        num_to_select = min(batch_size, X_candidates.shape[0])
        if num_to_select == 0:
            return np.array([])

        top_indices = np.argsort(acquisition_scores)[::-1][:num_to_select]
        selected_points = X_candidates[top_indices]
        return selected_points

    def _evaluate_points(self, func: Callable[[np.ndarray], np.ndarray], X_to_eval: np.ndarray) -> np.ndarray:
        # Evaluate the points in X.
        # On the first evaluation, infer M from func(x). Set self.n_obj = y.shape[0] and enforce this dimension in all later operations.
        # This method must be the only place where func is called.
        # Respect the remaining budget: do not exceed self.budget evaluations in total.
        # Use simple, clear looping over points, and clip points to the bounds if necessary.
        # Update self.n_evals by the actual number of function calls performed.
        # Clip points to self.bounds before calling func, to respect the problem domain.
        # lb, ub = self.bounds
        # X_clipped = np.clip(X, lb, ub)
        # then call func on X_clipped
        # Return an array of shape (n_points, n_obj).

        num_to_evaluate = min(X_to_eval.shape[0], self.budget - self.n_evals)
        if num_to_evaluate <= 0:
            return np.array([]) # No budget left

        X_subset = X_to_eval[:num_to_evaluate]
        
        # Clip points to self.bounds
        lb, ub = self.bounds[0], self.bounds[1]
        X_clipped = np.clip(X_subset, lb, ub)

        y_results = []
        for x_point in X_clipped:
            y_val = func(x_point)
            if self.n_obj is None:
                self.n_obj = len(y_val)
            elif len(y_val) != self.n_obj:
                raise ValueError(f"Function returned {len(y_val)} objectives, but expected {self.n_obj}.")
            y_results.append(y_val)
        
        self.n_evals += num_to_evaluate
        return np.array(y_results)

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        # Update the archive with new evaluations.
        # Do not change the function signature.
        # IMPORTANT: Keep ALL evaluated points in the archive (self.X, self.y) for surrogate model training.
        # The surrogate model needs both dominated and non-dominated points to learn the objective landscape.
        # Track the non-dominated (Pareto) front separately for returning results.
        # Dominance comparisons MUST use y.shape[1] == self.n_obj, never a fixed number of objectives.
        
        if new_X.shape[0] == 0: # Nothing new to add
            return

        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Main optimization loop.
        # func takes an array of shape (n_dims,) and returns np.ndarray of shape (M,) (one value per objective).
        # Always use _evaluate_points to call func so that the budget is respected.
        # The full run must complete within approximately 60 seconds.
        # Return a tuple (F_pareto, X_pareto), where F_pareto has shape (K, n_obj)
        # and X_pareto has shape (K, n_dims) for the final non-dominated set.
        # The algorithm MUST remain correct for any number of objectives self.n_obj >= 2 without code changes.

        # 1. Initial Design
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # 2. Main Optimization Loop
        while self.n_evals < self.budget:
            # We need enough points to train a BaggingRegressor and extract a Pareto front for acquisition.
            # If not enough points, continue with random sampling for initial exploration.
            if self.X.shape[0] < self.n_init: # Use n_init as a threshold for switching to model-based
                new_X_batch = self._sample_points(self.batch_size)
            else:
                self._fit_model(self.X, self.y)
                new_X_batch = self._select_next_points(self.batch_size)
            
            if new_X_batch.shape[0] == 0: # No points selected or budget exhausted
                break

            new_y_batch = self._evaluate_points(func, new_X_batch)
            if new_y_batch.shape[0] == 0: # Budget exhausted during evaluation
                break
            # [PATCH 2026-06-22] Budget-boundary off-by-one fix.
            # `_evaluate_points` caps the final batch at the remaining budget and
            # can return fewer rows than `new_X_batch`. The original line was:
            #     self._update_eval_points(new_X_batch, new_y_batch)
            # which vstacked the FULL new_X_batch against the truncated new_y_batch,
            # leaving self.X longer than self.y and crashing the Pareto extraction
            # (`self.X[pareto_indices]`) with an IndexError on the last batch.
            # Truncating X to the number of points actually evaluated keeps the
            # archive aligned; behaviour is otherwise unchanged (the desync only
            # ever occurred in the final batch, after which the loop exits).
            new_X_batch = new_X_batch[: new_y_batch.shape[0]]
            self._update_eval_points(new_X_batch, new_y_batch)

        # 3. Extract Pareto Front
        # Ensure there are evaluated points before trying to find Pareto front
        if self.y is None or self.y.shape[0] == 0:
            return np.array([]), np.array([])

        pareto_indices = is_pareto_efficient(self.y)
        F_pareto = self.y[pareto_indices]
        X_pareto = self.X[pareto_indices]

        return F_pareto, X_pareto

