from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


class MOBOESILimitedGP:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None,
                 n_initial_design: int | None = None, # Hyperparameter (default is budget-aware)
                 n_candidates_acquisition: int = 150, # Hyperparameter (reduced default)
                 model_window_size: int = 50, # Hyperparameter (tuned for speed)
                 n_acquisition_weights: int = 30, # Hyperparameter (reduced default)
                 gp_alpha: float = 1e-5): # Hyperparameter (noise level)
        
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
        # Choose a reasonable number of initial evaluations.
        # Use a small, budget-aware design (for example, proportional to dim, but not more than a fraction of the budget).
        if n_initial_design is None:
            # Default for n_initial_design based on problem statement guidelines
            # Capped at 50 to align with hyperparameter search range and prevent excessively large initial designs.
            self.n_initial_design = max(5, min(2 * dim + 1, budget // 4, 50))
        else:
            self.n_initial_design = n_initial_design

        self.n_candidates_acquisition = n_candidates_acquisition
        self.model_window_size = model_window_size
        self.n_acquisition_weights = n_acquisition_weights
        self.gp_alpha = gp_alpha
        self.gp_n_restarts_optimizer: int = 0 # Fixed to 0 for maximum speed during optimization

        # The number of objectives (self.n_obj) is unknown a priori.
        # It MUST be inferred on the first call to func inside _evaluate_points.
        self.n_obj: int | None = None

        # X has shape (n_points, n_dims), y has shape (n_points, n_obj)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.n_evals = 0  # the number of function evaluations

        # Store GP models
        self.models: list[GaussianProcessRegressor] = []

    def _sample_points(self, n_points: int) -> np.ndarray:
        """
        Samples n_points candidate points efficiently within self.bounds.
        Uses Sobol sequence for better space-filling properties.
        """
        if n_points <= 0:
            return np.array([]).reshape(0, self.dim)
        
        # Use scrambled Sobol sequence for better space-filling properties
        # Adding a seed from np.random for more varied samples across runs if desired,
        # but for reproducible HPO, a fixed seed might be considered.
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=np.random.randint(0, 2**30)) 
        
        # Generate samples in [0, 1]^dim
        samples_unit = sampler.random(n=n_points)
        
        # Scale samples to self.bounds
        lb = self.bounds[0]
        ub = self.bounds[1]
        samples_scaled = qmc.scale(samples_unit, lb, ub)
        return samples_scaled

    def _fit_model(self, X_data: np.ndarray, y_data: np.ndarray):
        """
        Fits Gaussian Process (GP) surrogate models for each objective.
        Applies a strict sliding window to keep model fitting tractable.
        """
        # Ensure there's enough data to fit the GP (at least 2 unique points for GP)
        if X_data.shape[0] < 2:
            return # Not enough data to fit GPs

        # Apply sliding window based on self.model_window_size
        window_size = self.model_window_size
        if X_data.shape[0] > window_size:
            X_window = X_data[-window_size:]
            y_window = y_data[-window_size:]
        else:
            X_window = X_data
            y_window = y_data
        
        # Initialize or update GP models for each objective
        # Re-initialize only if n_obj changes or models are not yet created
        if not self.models or len(self.models) != self.n_obj:
            self.models = []
            for _ in range(self.n_obj):
                # Using Matern kernel (nu=2.5) with automatic length_scale and WhiteKernel for noise.
                # Tight length_scale_bounds and noise_level_bounds for better stability/speed.
                kernel = Matern(length_scale=np.ones(self.dim), nu=2.5, length_scale_bounds=(1e-2, 1e2)) \
                         + WhiteKernel(noise_level=self.gp_alpha, noise_level_bounds=(1e-7, 1e-3))
                self.models.append(GaussianProcessRegressor(kernel=kernel,
                                                            alpha=self.gp_alpha, # Initial noise level
                                                            n_restarts_optimizer=self.gp_n_restarts_optimizer,
                                                            random_state=42)) # Fixed random state for reproducibility in tuning

        for i in range(self.n_obj):
            # Ensure y_window[:, i] has at least two unique values for GP fitting.
            # If an objective is constant, GP fitting will fail or be trivial.
            if len(np.unique(y_window[:, i])) < 2:
                # If an objective is constant within the window, skip fitting for this objective.
                # The model for this objective will retain its previous state or initial state.
                continue
            self.models[i].fit(X_window, y_window[:, i])

    def _generate_weights(self, n_weights: int) -> np.ndarray:
        """
        Generates n_weights vectors from a Dirichlet distribution,
        ensuring they sum to 1 and have a small epsilon to avoid zero weights.
        """
        if self.n_obj is None or self.n_obj < 1:
            return np.array([])
        
        weights = np.random.dirichlet(np.ones(self.n_obj), n_weights)
        # Ensure no weight is exactly zero, which can cause issues with Tchebycheff scalarization
        weights = np.maximum(weights, 1e-6)
        weights /= np.sum(weights, axis=1, keepdims=True) # Re-normalize after clipping
        return weights

    def _acquisition_function(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        Implements a multi-objective acquisition function (ParEGO-like Expected Scalarized Improvement).
        Calculates the acquisition function value for each point in X_candidates.
        """
        # Return uniform acquisition to encourage exploration if not enough data or models not fitted
        if self.X is None or self.y is None or self.X.shape[0] < 2 or self.n_obj is None or not self.models:
            return np.ones(X_candidates.shape[0]) * 1e-6 # Small positive value to avoid issues

        # 1. Dynamic Normalization based on current observed objective ranges
        ideal_point = np.min(self.y, axis=0)
        nadir_point = np.max(self.y, axis=0)
        ranges = nadir_point - ideal_point
        # Add a small epsilon to avoid division by zero for objectives with constant values
        ranges[ranges < 1e-9] = 1e-9

        # Normalize observed objectives in the archive
        y_norm_archive = (self.y - ideal_point) / ranges
        # Clamp normalized values to [0, 1] to prevent issues with GP predictions outside observed range
        y_norm_archive = np.clip(y_norm_archive, 0.0, 1.0)

        # 2. Predict means and standard deviations for X_candidates using GP models
        mu_sigma_list_for_each_obj = []
        for m in range(self.n_obj):
            # Predict uses the last fitted model state.
            mu_sigma_list_for_each_obj.append(self.models[m].predict(X_candidates, return_std=True))
            
        mu_preds = np.array([m[0] for m in mu_sigma_list_for_each_obj]).T  # Shape: (n_candidates, n_obj)
        sigma_preds = np.array([m[1] for m in mu_sigma_list_for_each_obj]).T  # Shape: (n_candidates, n_obj)

        # Normalize predicted means and stds using the same ideal/nadir/ranges
        mu_norm_preds = (mu_preds - ideal_point) / ranges
        sigma_norm_preds = sigma_preds / ranges
        # Clamp normalized values to [0, 1] for predictions, maintaining consistency in the normalized space
        mu_norm_preds = np.clip(mu_norm_preds, 0.0, 1.0)

        # 3. Generate random weight vectors for Tchebycheff scalarization
        weights = self._generate_weights(self.n_acquisition_weights)
        if weights.shape[0] == 0: # If n_obj is not set or invalid
             return np.ones(X_candidates.shape[0]) * 1e-6

        acq_values = np.zeros(X_candidates.shape[0])

        # 4. Calculate Expected Improvement (EI) for each weight vector and average
        for w in weights:
            # Calculate current best Tchebycheff value from archive for this weight vector
            # Tchebycheff for minimization: max_m (w_m * (y_norm_m - utopian_point_norm_m))
            # With utopian_point_norm_m implicitly 0 for normalized objectives.
            current_scalarized_y = np.max(w * y_norm_archive, axis=1)
            best_f_w = np.min(current_scalarized_y) # We seek to minimize Tchebycheff, so 'best' is minimum

            # Calculate predicted Tchebycheff mean and standard deviation for candidates
            # Approximation: Tchebycheff of the mean predictions, and for std, use max of weighted stds.
            # This is a common heuristic in ParEGO for computational efficiency.
            mu_scalarized_candidates = np.max(w * mu_norm_preds, axis=1)
            sigma_scalarized_candidates = np.max(w * sigma_norm_preds, axis=1)

            # Calculate EI for minimization.
            # EI = (best_f_w - mu_scalarized_candidates) * Phi(Z) + sigma_scalarized_candidates * phi(Z)
            
            # Suppress warnings for division by zero or invalid values (e.g., when sigma is 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                # Avoid division by zero if sigma_scalarized_candidates is very small
                Z = np.zeros_like(sigma_scalarized_candidates)
                non_zero_sigma_idx = sigma_scalarized_candidates > 1e-9
                Z[non_zero_sigma_idx] = (best_f_w - mu_scalarized_candidates[non_zero_sigma_idx]) / sigma_scalarized_candidates[non_zero_sigma_idx]
                
                ei = (best_f_w - mu_scalarized_candidates) * norm.cdf(Z) + sigma_scalarized_candidates * norm.pdf(Z)
                
                # Handle cases where sigma is essentially zero: EI should be 0.
                ei[~non_zero_sigma_idx] = 0.0
                
                # EI should always be non-negative.
                ei[ei < 0] = 0.0
            
            acq_values += ei

        acq_values /= len(weights) # Average EI over all weight vectors
        return acq_values

    def _select_next_points(self, batch_size: int) -> np.ndarray:
        """
        Selects the next points to evaluate using the acquisition function.
        Generates candidate points, scores them, and selects the best.
        """
        # Generate a large pool of candidate points for acquisition evaluation
        candidate_X = self._sample_points(self.n_candidates_acquisition)
        
        # Calculate acquisition values for all candidates
        acquisition_scores = self._acquisition_function(candidate_X)
        
        # Select the point(s) with the highest acquisition score(s)
        # For sequential BO, batch_size is typically 1.
        sorted_indices = np.argsort(acquisition_scores)[::-1] # Descending order
        selected_indices = sorted_indices[:batch_size]
        
        return candidate_X[selected_indices]

    def _evaluate_points(self, func: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> np.ndarray:
        """
        Evaluates the points in X using the black-box function `func`.
        Manages budget, infers n_obj, and clips points to bounds.
        """
        results_y = []
        actual_eval_count = 0
        for i in range(X.shape[0]):
            if self.n_evals >= self.budget:
                break # Stop if budget is exhausted
            
            # Clip points to self.bounds before calling func, to respect the problem domain.
            lb, ub = self.bounds[0], self.bounds[1]
            x_clipped = np.clip(X[i], lb, ub)
            
            y_val = func(x_clipped)
            
            # Infer n_obj on the first evaluation
            if self.n_obj is None:
                self.n_obj = y_val.shape[0]
                if self.n_obj < 2: # Ensure it's a multi-objective problem
                    raise ValueError(f"Multi-objective function must return at least 2 objectives, but got {self.n_obj}.")
            elif y_val.shape[0] != self.n_obj:
                raise ValueError(f"Function returned {y_val.shape[0]} objectives, but {self.n_obj} were expected.")
            
            results_y.append(y_val)
            self.n_evals += 1
            actual_eval_count += 1
        
        if not results_y: # If no evaluations were performed due to budget
            # Return an empty array of correct dimension if n_obj is known, else (0,0)
            return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0)

        return np.array(results_y)

    def _get_pareto_front(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Identifies the non-dominated (Pareto) front from the given points.
        Uses a simple O(N^2) comparison, suitable for typical budgets (N <= 500).
        """
        if X is None or X.shape[0] == 0:
            return np.array([]).reshape(0, self.dim), np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0)

        is_pareto = np.ones(y.shape[0], dtype=bool)
        for i in range(y.shape[0]):
            if not is_pareto[i]: # Skip if already dominated by an earlier point
                continue
            for j in range(y.shape[0]):
                if i == j:
                    continue
                # Point j dominates point i if all objectives of j are less than or equal to i,
                # AND at least one objective of j is strictly less than i (for minimization).
                if np.all(y[j] <= y[i]) and np.any(y[j] < y[i]):
                    is_pareto[i] = False
                    break
        return X[is_pareto], y[is_pareto]

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        """
        Updates the archive with new evaluations. All evaluated points are kept.
        """
        if new_X.shape[0] == 0: # No new points to add
            return

        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Main optimization loop for multi-objective black-box optimization.
        """
        # 1. Initial Design: Evaluate a set of points to start the optimization.
        initial_X = self._sample_points(self.n_initial_design)
        # Ensure at least one point is evaluated if budget allows, even if n_initial_design is 0 or very small
        if initial_X.shape[0] == 0 and self.budget > 0:
             initial_X = self._sample_points(1) 
        # Cap initial design if it exceeds the total budget
        if initial_X.shape[0] > self.budget:
            initial_X = initial_X[:self.budget]
        
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # 2. Main Bayesian Optimization Loop
        while self.n_evals < self.budget:
            # Ensure there's enough data for robust GP fitting (at least 2 points) and n_obj is known.
            if self.X.shape[0] < 2 or self.n_obj is None:
                # If not enough data for robust GP, add more random points to reach a basic threshold
                # This ensures we can fit GPs in the next iteration.
                num_to_add = max(1, 2 - self.X.shape[0]) 
                next_X_batch = self._sample_points(num_to_add)
            else:
                # Fit surrogate models using the current archive (or a window of it)
                self._fit_model(self.X, self.y)
                # Select next point(s) using the acquisition function (sequential batch_size=1)
                next_X_batch = self._select_next_points(batch_size=1)

            # Evaluate the selected point(s)
            new_y = self._evaluate_points(func, next_X_batch)
            if new_y.shape[0] == 0: # If budget ran out during evaluation, stop
                break
            
            # Update archive with new evaluations
            self._update_eval_points(next_X_batch, new_y)

        # 3. Return the final non-dominated front from the entire archive
        F_pareto, X_pareto = self._get_pareto_front(self.X, self.y)
        return F_pareto, X_pareto

