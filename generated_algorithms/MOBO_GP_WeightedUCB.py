from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
import scipy.linalg # For GP implementation

# Helper for non-dominated sort (for minimization)
def is_dominated(p1: np.ndarray, p2: np.ndarray) -> bool:
    # Returns True if p1 is dominated by p2 (for minimization)
    # i.e., p2 is strictly better in at least one objective and no worse in others.
    return np.all(p2 <= p1) and np.any(p2 < p1)

def get_pareto_front(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies the non-dominated (Pareto) front from a set of points.
    Assumes minimization for objectives.
    """
    if Y.shape[0] == 0:
        return np.array([]).reshape(0, Y.shape[1]), np.array([]).reshape(0, X.shape[1])
    
    is_nondominated = np.ones(Y.shape[0], dtype=bool)
    
    # O(N^2) complexity, but N is limited by budget (50-500), so acceptable for final processing.
    for i in range(Y.shape[0]):
        if not is_nondominated[i]:
            continue
        for j in range(Y.shape[0]):
            if i == j:
                continue
            if is_dominated(Y[i], Y[j]):
                is_nondominated[i] = False
                break
    
    return Y[is_nondominated], X[is_nondominated]

# Minimal Gaussian Process Regressor implementation
class _GaussianProcessRegressor:
    def __init__(self, length_scale: float, sigma_f: float, noise_std: float):
        self.length_scale = length_scale
        self.sigma_f = sigma_f # Fixed to 1.0 as objectives are normalized internally
        self.noise_std = noise_std 
        self.X_train = None
        self.y_train = None
        self.L_factor = None # Stores (L, lower) from cho_factor
        self.alpha_vec = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Squared Exponential (RBF) kernel
        # K(x_i, x_j) = sigma_f^2 * exp(-0.5 * ||x_i - x_j||^2 / length_scale^2)
        # Compute squared Euclidean distances efficiently
        sq_dist = (np.sum(X1**2, 1)[:, None] + np.sum(X2**2, 1)[None, :]) - 2 * X1 @ X2.T
        sq_dist = np.maximum(sq_dist, 0) # Ensure non-negative distances
        
        return self.sigma_f**2 * np.exp(-0.5 * sq_dist / (self.length_scale**2))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train.flatten() # Ensure y_train is 1D for single objective GP
        
        K = self._kernel(X_train, X_train)
        # Add noise for numerical stability (jitter)
        K_stable = K + (self.noise_std**2) * np.eye(X_train.shape[0]) # Use noise_std squared for variance
        
        try:
            # Cholesky decomposition for solving linear systems efficiently and stably
            self.L_factor = scipy.linalg.cho_factor(K_stable, lower=True)
            self.alpha_vec = scipy.linalg.cho_solve(self.L_factor, self.y_train)
        except scipy.linalg.LinAlgError:
            raise RuntimeError("Cholesky decomposition failed during GP fit. "
                               "This can happen with numerically unstable kernel parameters or duplicate points. "
                               "Consider adjusting length_scale, sigma_f, or increasing noise_std.")

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.X_train is None:
            raise RuntimeError("GP model not fitted.")
            
        K_star = self._kernel(self.X_train, X_test)
        
        # Mean prediction
        mu_star = K_star.T @ self.alpha_vec

        # Variance prediction: K_ss - K_s^T K_stable_inv K_s
        K_star_star = self._kernel(X_test, X_test)
        
        # K_stable_inv_K_star computes K_stable^-1 @ K_star
        K_stable_inv_K_star = scipy.linalg.cho_solve(self.L_factor, K_star)
        
        # Variance term: K_star.T @ K_stable_inv_K_star. We need its diagonal.
        var_star = np.diag(K_star_star - K_star.T @ K_stable_inv_K_star)
        
        var_star = np.maximum(0, var_star) # Ensure non-negative variance
        
        return mu_star, np.sqrt(var_star)


class MOBO_GP_WeightedUCB:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None,
                 n_init: int = None, n_model_train: int = 50, n_candidates: int = 200,
                 n_weights: int = 20, length_scale: float = 1.0, 
                 beta: float = 2.0, noise_std: float = 1e-6):
        # Fixed problem parameters
        self.budget = budget
        self.dim = dim
        if bounds is None:
            self.bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        else:
            self.bounds = np.asarray(bounds, dtype=float)

        # Hyperparameters (tuned by SMAC)
        # Default for n_init is budget-aware, but config space will use a fixed range.
        if n_init is None:
            self.n_init = min(max(2 * dim + 1, 10), budget // 4)
        else:
            self.n_init = n_init
        
        self.n_model_train = n_model_train # Sliding window size for GP training
        self.n_candidates = n_candidates   # Number of candidate points for acquisition evaluation
        self.n_weights = n_weights         # Number of weight vectors for scalarization
        self.length_scale = length_scale   # GP kernel length scale
        self.sigma_f = 1.0                 # GP output scale (fixed for normalized objectives)
        self.beta = beta                   # Exploration-exploitation trade-off for UCB
        self.noise_std = noise_std         # GP noise standard deviation

        # The number of objectives (self.n_obj) is unknown a priori.
        self.n_obj: int | None = None

        # X has shape (n_points, n_dims), y has shape (n_points, n_obj)
        # Store ALL evaluated points (negated objectives for internal minimization)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None 
        self.n_evals = 0  # the number of function evaluations
        
        # List of GP models, one for each objective
        self.gps: list[_GaussianProcessRegressor] = []

    def _sample_points(self, n_points: int) -> np.ndarray:
        # Sample n_points candidate points efficiently within self.bounds using Sobol sequence
        sampler = qmc.Sobol(d=self.dim)
        # Generate samples in [0, 1]^dim
        unit_samples = sampler.random(n=n_points)
        
        # Scale samples to self.bounds
        lower_bounds = self.bounds[0]
        upper_bounds = self.bounds[1]
        scaled_samples = lower_bounds + unit_samples * (upper_bounds - lower_bounds)
        return scaled_samples

    def _normalize_objectives(self, y: np.ndarray, min_y: np.ndarray, max_y: np.ndarray) -> np.ndarray:
        # Normalize objectives to [0, 1] range based on observed min/max.
        # This is applied to negated objectives, so 0 is "best", 1 is "worst".
        # min_y and max_y should be (1, n_obj) or (n_obj,)
        min_y = min_y.reshape(1, -1)
        max_y = max_y.reshape(1, -1)
        
        epsilon = 1e-6 # For numerical stability to avoid division by zero
        
        # Clip y values to be within [min_y, max_y] to handle GP predictions
        # that might fall outside observed range.
        y_clipped = np.clip(y, min_y, max_y)
        
        range_y = max_y - min_y + epsilon
        y_norm = (y_clipped - min_y) / range_y
        return y_norm

    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray):
        # Fit a surrogate model on (X_train, y_train).
        # y_train here is already negated.
        
        if self.n_obj is None:
            raise RuntimeError("Number of objectives (n_obj) not inferred yet.")
            
        # Initialize GP models if not already done or if n_obj changed (shouldn't happen)
        if not self.gps or len(self.gps) != self.n_obj:
            self.gps = [
                _GaussianProcessRegressor(
                    length_scale=self.length_scale, 
                    sigma_f=self.sigma_f, # Fixed to 1.0
                    noise_std=self.noise_std 
                ) for _ in range(self.n_obj)
            ]
        
        # Fit each GP model for each objective
        for i in range(self.n_obj):
            self.gps[i].fit(X_train, y_train[:, i])

    def _acquisition_function(self, X_cand: np.ndarray) -> np.ndarray:
        # Implement a multi-objective acquisition function (Weighted UCB)
        # X_cand: candidate points (n_candidates, dim)
        # Returns: 1-D score per candidate of shape (n_candidates,)
        
        if not self.gps or self.gps[0].X_train is None:
            # If models are not fitted (e.g., initial phase), return random scores for exploration
            return np.random.rand(X_cand.shape[0])
            
        n_candidates = X_cand.shape[0]
        
        # Get overall min_y and max_y from the entire archive for normalization
        # self.y contains negated objectives.
        # min_y_archive: vector of most negative (best original) observed values.
        # max_y_archive: vector of least negative (worst original) observed values.
        min_y_archive = np.min(self.y, axis=0)
        max_y_archive = np.max(self.y, axis=0)
        
        # Predict means and standard deviations for all objectives for candidate points
        mu_preds = np.zeros((n_candidates, self.n_obj))
        sigma_preds = np.zeros((n_candidates, self.n_obj))
        
        for i in range(self.n_obj):
            mu_preds[:, i], sigma_preds[:, i] = self.gps[i].predict(X_cand)
        
        # Normalize predicted means using the archive's min/max.
        # A normalized negated objective of 0 means best original value, 1 means worst.
        mu_preds_norm = self._normalize_objectives(mu_preds, min_y_archive, max_y_archive)
        
        # Heuristic normalization for sigma: scale by objective range
        # Ensures that uncertainty is considered relative to the observed scale of each objective.
        sigma_preds_norm = sigma_preds / (max_y_archive - min_y_archive + 1e-6) 
        
        # Generate weight vectors for scalarization.
        # Use a simplex for weights to ensure sum to 1.
        if self.n_obj == 2: # Special case for 2 objectives (weights sum to 1)
            weights_unit_interval = np.linspace(0, 1, self.n_weights).reshape(-1, 1)
            weights = np.hstack([weights_unit_interval, 1 - weights_unit_interval])
        else:
            # Use Sobol sequence on (n_obj - 1) dimensions to generate diverse weights on a simplex.
            # This is a common method for generating quasi-random points on a simplex.
            sampler_weights = qmc.Sobol(d=self.n_obj - 1)
            unit_weights = sampler_weights.random(n=self.n_weights)
            
            # Transform to simplex using cumulative sums of differences
            temp_weights = np.sort(np.hstack([np.zeros((self.n_weights, 1)), unit_weights, np.ones((self.n_weights, 1))]), axis=1)
            weights = np.diff(temp_weights, axis=1)

        acquisition_scores = np.zeros(n_candidates)
        
        for i in range(n_candidates):
            current_max_ucb = -np.inf # Initialize with negative infinity
            
            for w in weights:
                # Scalarized mean: weighted sum of normalized negated objective means
                scalarized_mu = np.sum(w * mu_preds_norm[i])
                
                # Scalarized uncertainty: weighted sum of normalized std dev (heuristic)
                # Assumes independence for simplicity; variance of sum of independent vars is sum of vars
                scalarized_sigma = np.sqrt(np.sum(w**2 * sigma_preds_norm[i]**2))
                
                # UCB acquisition: Maximize (-mean + beta * sigma)
                # Since mu_preds_norm=0 is best, we want to maximize -scalarized_mu.
                # High sigma contributes positively, promoting exploration.
                ucb_score = -scalarized_mu + self.beta * scalarized_sigma
                
                if ucb_score > current_max_ucb:
                    current_max_ucb = ucb_score
            
            acquisition_scores[i] = current_max_ucb
            
        return acquisition_scores

    def _select_next_points(self, batch_size: int) -> np.ndarray:
        # Generate a large pool of candidate points using Sobol sequence
        X_cand = self._sample_points(self.n_candidates)
        
        # Calculate acquisition scores for candidates
        acquisition_scores = self._acquisition_function(X_cand)
        
        # Select batch_size candidates with the highest scores
        # Use argsort to get indices of top scores
        top_indices = np.argsort(acquisition_scores)[::-1][:batch_size]
        
        return X_cand[top_indices]

    def _evaluate_points(self, func: Callable[[np.ndarray], np.ndarray], X_to_eval: np.ndarray) -> np.ndarray:
        # Evaluate the points in X_to_eval.
        # This method must be the only place where func is called.
        
        if self.n_evals >= self.budget:
            return np.array([]) # No budget left
            
        num_points = X_to_eval.shape[0]
        actual_evals = min(num_points, self.budget - self.n_evals)
        
        if actual_evals == 0:
            return np.array([])

        y_results = []
        for i in range(actual_evals):
            x_point = X_to_eval[i]
            
            # Clip points to self.bounds before calling func
            lb, ub = self.bounds[0], self.bounds[1]
            x_clipped = np.clip(x_point, lb, ub)
            
            # Call the black-box function
            y_raw = func(x_clipped)
            
            # Infer M (n_obj) from the first evaluation
            if self.n_obj is None:
                self.n_obj = y_raw.shape[0]
                if self.n_obj < 2:
                    raise ValueError(f"Multi-objective function must return at least 2 objectives, but got {self.n_obj}.")

            # Ensure consistency of objective dimension
            if y_raw.shape[0] != self.n_obj:
                raise ValueError(f"Function returned {y_raw.shape[0]} objectives, but {self.n_obj} were expected.")
            
            # Negate objectives for internal minimization logic (as we want to maximize Hypervolume)
            y_processed = -y_raw
            y_results.append(y_processed)
            
        self.n_evals += actual_evals
        
        if not y_results:
            return np.array([])
            
        return np.array(y_results)

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        # Update the archive with new evaluations.
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])


    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Initial design phase (Latin Hypercube Sampling)
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Main optimization loop
        # We process one point at a time (batch_size=1) for sequential Bayesian Optimization
        batch_size = 1 

        while self.n_evals < self.budget:
            # Select the most recent `n_model_train` points for GP training
            # This implements the sliding window, crucial for controlling GP complexity (O(N_train^3)).
            if self.X.shape[0] > self.n_model_train:
                X_train_window = self.X[-self.n_model_train:]
                y_train_window = self.y[-self.n_model_train:]
            else:
                X_train_window = self.X
                y_train_window = self.y
            
            # Fit GP models
            try:
                self._fit_model(X_train_window, y_train_window)
            except RuntimeError as e:
                # If GP fitting fails, fall back to random sampling to avoid crashing
                print(f"Warning: GP model fitting failed ({e}). Falling back to random sampling for this iteration.")
                X_next = self._sample_points(batch_size)
            else:
                # Select next points using acquisition function
                X_next = self._select_next_points(batch_size)
            
            # Evaluate new points
            y_next = self._evaluate_points(func, X_next)
            
            # If no points were evaluated (budget exhausted), break
            if y_next.shape[0] == 0:
                break
            
            # Update archive
            self._update_eval_points(X_next, y_next)

        # Final Pareto front extraction (objectives are still negated)
        F_pareto_negated, X_pareto = get_pareto_front(self.X, self.y)
        
        # Denegate objectives for final output (return original maximized values)
        F_pareto = -F_pareto_negated

        return F_pareto, X_pareto

