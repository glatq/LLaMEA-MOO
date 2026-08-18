from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress sklearn warnings about convergence if necessary, or numerical issues
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class MOBORefinedTchebyLCBGP:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None,
                 batch_size: int = 4,
                 n_candidates_acq: int = 100,
                 max_train_size: int = 50,
                 n_weight_vectors: int = 50,
                 kappa: float = 2.0,
                 jitter: float = 1e-6,
                 n_init: int = 10 # Default for n_init hyperparameter
                 ):
        # Fixed problem parameters
        self.budget = budget
        self.dim = dim
        # bounds has shape (2, dim), bounds[0]: lower bound, bounds[1]: upper bound
        # The environment (evaluator / problem provider) should pass the true bounds.
        # Do NOT overwrite self.bounds with hard-coded values.
        if bounds is None:
            # Fallback: assume a simple [0.0, 1.0]^dim box if no bounds are provided.
            self.bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        else:
            self.bounds = np.asarray(bounds, dtype=float)

        # Hyperparameters (tuned by SMAC, defined in Space)
        self.batch_size = batch_size
        self.n_candidates_acq = n_candidates_acq
        self.max_train_size = max_train_size
        self.n_weight_vectors = n_weight_vectors
        self.kappa = kappa
        self.jitter = jitter
        self.n_init = n_init # Number of initial evaluations as provided by SMAC or default

        # The number of objectives (self.n_obj) is unknown a priori.
        # It MUST be inferred on the first call to func inside _evaluate_points.
        self.n_obj: int | None = None

        # X_archive stores points scaled to [0,1]^dim, y_archive stores original objective values.
        # X_archive has shape (n_points, n_dims), y_archive has shape (n_points, n_obj)
        self.X_archive: np.ndarray | None = None
        self.y_archive: np.ndarray | None = None
        self.n_evals = 0  # the number of function evaluations

        # Pareto front storage (calculated at the end)
        self.X_pareto: np.ndarray | None = None
        self.y_pareto: np.ndarray | None = None

        # Surrogate models
        self.gp_models: list[GaussianProcessRegressor] = []

        # Objective normalization parameters (dynamically updated)
        self.ideal_point: np.ndarray | None = None
        self.nadir_point: np.ndarray | None = None

        # Scaler for input X (to [0,1] range)
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit scaler once based on bounds. self.bounds is (2, dim).
        # MinMaxScaler expects (n_samples, n_features). Here, 2 samples (min/max), dim features.
        # This correctly sets the min/max for each of the 'dim' features.
        self.x_scaler.fit(self.bounds) 

        # Weight vectors for Tchebycheff scalarization
        self.weights: np.ndarray | None = None

    def _sample_points(self, n_points: int) -> np.ndarray:
        # Sample n_points candidate points efficiently within self.bounds.
        # Use Sobol sequence for quasi-random sampling in [0, 1]^dim
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # The 'random' method guarantees 'n_points' samples.
        sample = sampler.random(n_points) # sample in [0, 1]^dim
        return sample

    def _generate_weights(self, n_vectors: int, n_objectives: int) -> np.ndarray:
        # Generates n_vectors weight vectors that sum to 1, using a Dirichlet distribution.
        # Each weight vector has length n_objectives.
        # Adding a small alpha (e.g., 0.5) ensures diversity and non-zero weights for all objectives.
        # For MOBO, diversity is often preferred, so using current np.random state is acceptable.
        return np.random.dirichlet([0.5] * n_objectives, n_vectors)

    def _fit_model(self, X_train_scaled: np.ndarray, y_train_normalized: np.ndarray):
        # Fit a surrogate model on (X_train_scaled, y_train_normalized).
        # X_train_scaled are points in [0,1]^dim.
        # y_train_normalized are objectives normalized to [0,1]^n_obj.
        # Store the fitted model on self.gp_models.
        
        # Need at least dim+1 points for GP fitting, plus one for variance
        if X_train_scaled.shape[0] < self.dim + 2: 
            # If not enough points, clear models to prevent using invalid ones
            self.gp_models = []
            return

        # Define a Matern kernel for each GP
        # Allow ConstantKernel (output scale) to be learned, with bounds.
        # n_restarts_optimizer=1 for speed, as per runtime constraints.
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(self.dim), length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(self.jitter, noise_level_bounds=(1e-8, 1e-3))
        
        self.gp_models = []
        for i in range(self.n_obj):
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, # Alpha is additional diagonal noise, WhiteKernel already handles observation noise
                                          n_restarts_optimizer=1, random_state=42)
            gp.fit(X_train_scaled, y_train_normalized[:, i])
            self.gp_models.append(gp)

    def _acquisition_function(self, X_candidates_scaled: np.ndarray) -> np.ndarray:
        # Implement a multi-objective acquisition function (Tchebycheff LCB).
        # X_candidates_scaled are points in [0,1]^dim.
        # Return a 1-D score per candidate of shape (n_points,) for selection.
        
        if not self.gp_models or X_candidates_scaled.shape[0] == 0: # No models fitted yet or no candidates
            # Return random scores to encourage exploration if models are not ready
            # Scores for minimization, so lower is better.
            return np.random.rand(X_candidates_scaled.shape[0])

        # Predict for all candidates and all objectives in one go (vectorized)
        mu_preds_norm_all_obj = np.zeros((X_candidates_scaled.shape[0], self.n_obj))
        sigma_preds_norm_all_obj = np.zeros((X_candidates_scaled.shape[0], self.n_obj))

        for obj_idx in range(self.n_obj):
            mu_obj, sigma_obj = self.gp_models[obj_idx].predict(X_candidates_scaled, return_std=True)
            mu_preds_norm_all_obj[:, obj_idx] = mu_obj
            # Add a small epsilon to sigma to prevent division by zero or numerical issues
            sigma_preds_norm_all_obj[:, obj_idx] = np.maximum(sigma_obj, 1e-10)

        # Generate weight vectors once if not already generated or if n_obj changed
        if self.weights is None or self.weights.shape[1] != self.n_obj:
            self.weights = self._generate_weights(self.n_weight_vectors, self.n_obj)

        # Calculate LCB for all candidates and all objectives
        # LCB is mu - kappa * sigma (for minimization)
        lcb_values_per_candidate_obj = mu_preds_norm_all_obj - self.kappa * sigma_preds_norm_all_obj # Shape (n_candidates_acq, n_obj)

        # Expand dimensions for broadcasting to compute weighted Tchebycheff
        # lcb_values_per_candidate_obj: (n_candidates_acq, 1, n_obj)
        # self.weights: (1, n_weight_vectors, n_obj)
        # Result of multiplication: (n_candidates_acq, n_weight_vectors, n_obj)
        weighted_lcb = self.weights[np.newaxis, :, :] * lcb_values_per_candidate_obj[:, np.newaxis, :]

        # Tchebycheff aggregation for minimization: max(w_i * (f_i - z_i^*))
        # Here, f_i is the LCB prediction, and z_i^* is 0 (for normalized objectives).
        # Result of np.max over axis=2: (n_candidates_acq, n_weight_vectors)
        tchebycheff_scores = np.max(weighted_lcb, axis=2)

        # The acquisition score for each candidate is the minimum (best) Tchebycheff LCB value
        # across all weight vectors. We are minimizing, so lower score is better.
        # Result of np.min over axis=1: (n_candidates_acq,)
        acquisition_scores = np.min(tchebycheff_scores, axis=1)
            
        return acquisition_scores

    def _select_next_points(self, batch_size: int) -> np.ndarray:
        # Select the next points to evaluate using the acquisition function.
        # Generate candidate points and score them with the acquisition function.
        # Return an array of shape (num_to_select, n_dims).
        
        # Generate candidates in [0,1]^dim
        X_candidates_scaled = self._sample_points(self.n_candidates_acq)
        
        # Score candidates using the acquisition function
        acquisition_scores = self._acquisition_function(X_candidates_scaled)
        
        # Select batch_size points with the lowest acquisition scores (since we minimize LCB)
        # Ensure we don't select more points than available candidates or remaining budget
        num_to_select = min(batch_size, X_candidates_scaled.shape[0], self.budget - self.n_evals)
        
        if num_to_select <= 0:
            return np.array([]).reshape(0, self.dim)

        # Get indices of the top 'num_to_select' candidates
        best_indices = np.argsort(acquisition_scores)[:num_to_select]
        
        return X_candidates_scaled[best_indices]

    def _evaluate_points(self, func: Callable[[np.ndarray], np.ndarray], X_scaled_to_eval: np.ndarray) -> np.ndarray:
        # Evaluate the points in X_scaled_to_eval (which are in [0,1]^dim).
        # This method must be the only place where func is called.
        # Update self.n_evals by the actual number of function calls performed.
        # Clip points to self.bounds before calling func, to respect the problem domain.
        # Return an array of shape (n_points, n_obj).

        if X_scaled_to_eval.shape[0] == 0:
            return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0)

        # Scale X_scaled_to_eval (in [0,1]^dim) to original bounds
        X_original_scale = self.x_scaler.inverse_transform(X_scaled_to_eval)
        
        # Clip points to self.bounds
        lb, ub = self.bounds[0], self.bounds[1]
        X_clipped = np.clip(X_original_scale, lb, ub)

        # Handle n_obj inference. This block runs only on the very first evaluation batch.
        if self.n_obj is None:
            # Evaluate the first point to infer n_obj
            first_y_val = func(X_clipped[0])
            if not isinstance(first_y_val, np.ndarray):
                first_y_val = np.asarray(first_y_val)
            self.n_obj = first_y_val.shape[0]
            self.n_evals += 1 # Count this first evaluation
            
            # Initialize new_y with the first evaluated point
            new_y = np.zeros((X_clipped.shape[0], self.n_obj))
            new_y[0] = first_y_val
            
            # Evaluate remaining points in the batch
            start_idx = 1
        else:
            new_y = np.zeros((X_clipped.shape[0], self.n_obj))
            start_idx = 0

        for i in range(start_idx, X_clipped.shape[0]):
            y_val = func(X_clipped[i])
            if not isinstance(y_val, np.ndarray):
                y_val = np.asarray(y_val)
            if y_val.shape[0] != self.n_obj:
                raise ValueError(f"Function returned {y_val.shape[0]} objectives, expected {self.n_obj}.")
            new_y[i] = y_val
            self.n_evals += 1
        
        return new_y

    def _update_eval_points(self, new_X_scaled: np.ndarray, new_y: np.ndarray):
        # Update the archive with new evaluations.
        # Store all evaluated points (X_archive in [0,1]^dim, y_archive in original scale).
        # Pareto front is calculated only at the end of the optimization run.
        
        if new_X_scaled.shape[0] == 0:
            return

        if self.X_archive is None:
            self.X_archive = new_X_scaled
            self.y_archive = new_y
        else:
            self.X_archive = np.vstack((self.X_archive, new_X_scaled))
            self.y_archive = np.vstack((self.y_archive, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Main optimization loop.
        
        # Initial Design Phase
        # Calculate the actual number of initial points to evaluate.
        # Ensure at least dim + 2 points for GP fitting (a bit more than theoretical min dim+1).
        # Also cap by the total budget.
        actual_n_init = min(self.n_init, self.budget)
        actual_n_init = max(actual_n_init, min(self.budget, self.dim + 2)) 

        if actual_n_init > 0:
            initial_X_scaled = self._sample_points(actual_n_init)
            initial_y = self._evaluate_points(func, initial_X_scaled)
            self._update_eval_points(initial_X_scaled, initial_y)

        # Optimization Loop
        while self.n_evals < self.budget:
            # Ensure enough data for GP fitting
            # Minimum required points for GP fitting is typically dim + 1
            if self.X_archive is None or self.X_archive.shape[0] < self.dim + 2:
                # If not enough data, just do random sampling to gather more points
                num_random_points = min(self.batch_size, self.budget - self.n_evals)
                if num_random_points <= 0: break # Budget exhausted
                next_X_scaled = self._sample_points(num_random_points)
                new_y = self._evaluate_points(func, next_X_scaled)
                self._update_eval_points(next_X_scaled, new_y)
                continue

            # Select data for GP training (sliding window)
            current_train_size = min(self.X_archive.shape[0], self.max_train_size)
            X_train = self.X_archive[-current_train_size:]
            y_train = self.y_archive[-current_train_size:]

            # Dynamic Global Objective Normalization: Use full archive to calculate ideal/nadir points
            self.ideal_point = np.min(self.y_archive, axis=0)
            self.nadir_point = np.max(self.y_archive, axis=0)
            
            # Add a small epsilon to prevent division by zero if min==max for an objective
            denom = self.nadir_point - self.ideal_point
            # If an objective is constant, its range is 0. Set denominator to epsilon.
            denom[denom == 0] = 1e-6 
            
            # Normalize the training data (y_train) using the global ideal/nadir points
            y_train_normalized = (y_train - self.ideal_point) / denom

            # Fit surrogate models
            self._fit_model(X_train, y_train_normalized)

            # Select next points to evaluate using the acquisition function
            next_X_scaled = self._select_next_points(self.batch_size)
            
            if next_X_scaled.shape[0] == 0: # No more points to evaluate (e.g. budget exhausted in _select_next_points)
                break

            # Evaluate selected points
            new_y = self._evaluate_points(func, next_X_scaled)
            
            # Update archive
            self._update_eval_points(next_X_scaled, new_y)

        # Calculate Pareto front only at the end of the optimization run
        if self.y_archive is None or self.y_archive.shape[0] == 0:
            F_pareto = np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0)
            X_pareto = np.array([]).reshape(0, self.dim)
        else:
            # Efficiently compute non-dominated set (assuming minimization)
            # This is an O(N^2 * M) operation, so it's done only once at the end.
            is_dominated = np.zeros(self.y_archive.shape[0], dtype=bool)
            for i in range(self.y_archive.shape[0]):
                for j in range(self.y_archive.shape[0]):
                    if i == j:
                        continue
                    # Check if point j dominates point i (all objectives of j <= i, and at least one objective of j < i)
                    if np.all(self.y_archive[j] <= self.y_archive[i]) and np.any(self.y_archive[j] < self.y_archive[i]):
                        is_dominated[i] = True
                        break
            
            self.X_pareto = self.X_archive[~is_dominated]
            self.y_pareto = self.y_archive[~is_dominated]

            # Scale X_pareto back to original bounds for return
            F_pareto = self.y_pareto
            X_pareto = self.x_scaler.inverse_transform(self.X_pareto)

        return F_pareto, X_pareto

