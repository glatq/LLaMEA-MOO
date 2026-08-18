from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import warnings

# Suppress ConvergenceWarning from sklearn.gaussian_process
warnings.filterwarnings("ignore", category=UserWarning)


class MOBO_GP_Tchebycheff_Refined:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None,
                 n_initial_design_points: int | None = None,
                 n_candidates_per_iteration: int = 300,  # Default adjusted
                 batch_size: int = 15,  # Default adjusted
                 sliding_window_size: int = 150,  # Default adjusted
                 num_weights: int = 70,  # Default adjusted
                 alpha_noise: float = 1e-7,  # Default adjusted, log-scaled range
                 kappa: float = 2.0,  # Exploration parameter, default value
                 model_restarts: int = 3,  # New tunable hyperparameter for GP kernel optimization
                 kernel_nu: float = 1.5):  # Fixed for simplicity and speed
        # Fixed problem parameters
        self.budget = budget
        self.dim = dim
        # bounds has shape (2, dim), bounds[0]: lower bound, bounds[1]: upper bound
        if bounds is None:
            # Fallback: assume a simple [0, 1]^dim box if no bounds are provided.
            self.bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        else:
            self.bounds = np.asarray(bounds, dtype=float)

        # Hyperparameters (tuned by SMAC, defined in Space)
        if n_initial_design_points is None:
            # Default strategy: at least max(dim+1, 5) points, but not more than 1/4 of budget.
            # Ensure at least 1 point is always sampled.
            self.n_init = max(1, min(max(self.dim + 1, 5), self.budget // 4))
        else:
            self.n_init = n_initial_design_points

        self.n_candidates_per_iteration = n_candidates_per_iteration
        self.batch_size = batch_size
        self.sliding_window_size = sliding_window_size
        self.num_weights = num_weights
        self.alpha_noise = alpha_noise
        self.kappa = kappa  # Exploration parameter
        self.model_restarts = model_restarts # Tunable number of restarts for kernel optimization
        self.kernel_nu = kernel_nu  # Fixed to 1.5

        # The number of objectives (self.n_obj) is unknown a priori.
        # It MUST be inferred on the first call to func inside _evaluate_points.
        self.n_obj: int | None = None

        # X has shape (n_points, n_dims), y has shape (n_points, n_obj)
        self.X: np.ndarray | None = None  # All evaluated design points
        self.y: np.ndarray | None = None  # All evaluated objective values
        self.n_evals = 0  # the number of function evaluations

        # Surrogate models
        self.gpr_models: list[GaussianProcessRegressor] = []

    def _sample_points(self, n_points: int) -> np.ndarray:
        # Sample n_points candidate points efficiently within self.bounds.
        # Use self.bounds[0] as lower bounds and self.bounds[1] as upper bounds.
        # Return array of shape (n_points, n_dims).

        # Use Sobol sequence for space-filling sampling
        # Added seed for reproducibility in sampling if needed, otherwise removed for true randomness
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=np.random.randint(0, 10000))
        samples = sampler.random(n=n_points)

        # Scale samples to the actual bounds
        lb = self.bounds[0]
        ub = self.bounds[1]
        scaled_samples = qmc.scale(samples, lb, ub)
        return scaled_samples

    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray):
        # Fit a surrogate model on (X, y).
        # Select data for sliding window
        if X_train.shape[0] > self.sliding_window_size:
            # Take the most recent points
            X_train_window = X_train[-self.sliding_window_size:]
            y_train_window = y_train[-self.sliding_window_size:]
        else:
            X_train_window = X_train
            y_train_window = y_train

        self.gpr_models = []
        # Fit one GPR for each objective
        for i in range(self.n_obj):
            # Define kernel: Matern with isotropic length scale and fixed nu.
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=self.kernel_nu)
            gpr = GaussianProcessRegressor(kernel=kernel,
                                           alpha=self.alpha_noise,  # Fixed noise for stability and speed
                                           n_restarts_optimizer=self.model_restarts, # Tunable restarts
                                           normalize_y=True,  # Added for numerical stability and implicit output scaling
                                           random_state=i)  # Ensure different random states for each GP
            gpr.fit(X_train_window, y_train_window[:, i])
            self.gpr_models.append(gpr)

    def _generate_weights(self) -> np.ndarray:
        # Generate self.num_weights weight vectors for Tchebycheff scalarization
        if self.n_obj == 1:
            return np.array([[1.0]])
        # Use Dirichlet distribution to generate diverse weights on the simplex
        rng = np.random.default_rng(seed=42)  # Using default_rng for better random number generation
        weights = rng.dirichlet(np.ones(self.n_obj), self.num_weights)
        return weights


    def _acquisition_function(self, X_candidates: np.ndarray) -> np.ndarray:
        # Implement a multi-objective acquisition function using Exploratory Tchebycheff.
        # Calculate the acquisition function value for each point in X_candidates.

        if not self.gpr_models or self.y.shape[0] < max(self.dim + 1, 2):
            # If models are not fitted yet or not enough data for robust ideal/nadir, return random scores
            return np.random.rand(X_candidates.shape[0])

        # Predict mean and standard deviation for all objectives for all candidates
        mu_list = []
        sigma_list = []
        for gpr in self.gpr_models:
            mu_obj, sigma_obj = gpr.predict(X_candidates, return_std=True)
            mu_list.append(mu_obj)
            sigma_list.append(sigma_obj)

        mu_candidates = np.array(mu_list).T  # Shape: (n_candidates, n_obj)
        sigma_candidates = np.array(sigma_list).T  # Shape: (n_candidates, n_obj)

        # Dynamic normalization based on current observed objective ranges
        # Use all evaluated points to determine ideal and nadir points
        ideal_point = np.min(self.y, axis=0)
        nadir_point = np.max(self.y, axis=0)

        # Avoid division by zero if an objective has constant values
        range_obj = nadir_point - ideal_point
        range_obj[range_obj < 1e-6] = 1e-6  # Small epsilon for stability

        # Normalize predicted means
        mu_candidates_norm = (mu_candidates - ideal_point) / range_obj
        # Normalize predicted standard deviations (spread is proportional to range)
        sigma_candidates_norm = sigma_candidates / range_obj

        # Generate weight vectors
        weights = self._generate_weights()  # shape (num_weights, n_obj)

        # Calculate the "effective" objective values for Tchebycheff, incorporating exploration (LCB-like)
        # We want to minimize (mu_k_norm - kappa * sigma_k_norm) for each objective.
        # This promotes selecting points where the mean is low AND/OR uncertainty is high.
        effective_objectives = mu_candidates_norm[:, None, :] - self.kappa * sigma_candidates_norm[:, None, :]

        # Vectorized Tchebycheff calculation:
        # weighted_effective_objectives: (n_candidates, num_weights, n_obj)
        weighted_effective_objectives = weights[None, :, :] * effective_objectives

        # Max over objectives for each candidate and each weight vector
        tchebycheff_vals_per_weight = np.max(weighted_effective_objectives, axis=2)  # Shape: (n_candidates, num_weights)

        # Acquisition score is the negative minimum Tchebycheff value across weights
        # Maximizing this means minimizing the actual Tchebycheff value.
        acquisition_scores = -np.min(tchebycheff_vals_per_weight, axis=1)  # Shape: (n_candidates,)

        return acquisition_scores

    def _select_next_points(self, batch_size: int) -> np.ndarray:
        # Select the next points to evaluate using the acquisition function.
        # Generate candidate points and score them with the acquisition function.
        # Return an array of shape (batch_size, n_dims).

        # Generate a large pool of candidate points
        X_candidates = self._sample_points(self.n_candidates_per_iteration)

        # Evaluate acquisition function for all candidates
        acquisition_scores = self._acquisition_function(X_candidates)

        # Select the best batch_size candidates (highest acquisition score)
        # Use argsort to get indices in descending order of acquisition score
        best_indices = np.argsort(acquisition_scores)[::-1][:batch_size]

        return X_candidates[best_indices]

    def _evaluate_points(self, func: Callable[[np.ndarray], np.ndarray], X_to_evaluate: np.ndarray) -> np.ndarray:
        # Evaluate the points in X_to_evaluate.
        # This method must be the only place where func is called.
        # Respect the remaining budget: do not exceed self.budget evaluations in total.
        # Update self.n_evals by the actual number of function calls performed.
        # Clip points to self.bounds before calling func, to respect the problem domain.

        if self.n_evals >= self.budget or X_to_evaluate.shape[0] == 0:
            return np.array([])  # No budget left or no points to evaluate

        n_points_to_eval = X_to_evaluate.shape[0]

        # Clip points to self.bounds
        lb = self.bounds[0]
        ub = self.bounds[1]
        X_clipped = np.clip(X_to_evaluate, lb, ub)

        y_evaluated_list = []
        for i in range(n_points_to_eval):
            if self.n_evals >= self.budget:
                break

            y_val = func(X_clipped[i])

            # Ensure y_val is a 1D numpy array, handling scalar returns for M=1
            if isinstance(y_val, (int, float)):
                y_val = np.array([y_val])
            elif not isinstance(y_val, np.ndarray):
                y_val = np.asarray(y_val)

            # Infer n_obj on the first evaluation
            if self.n_obj is None:
                self.n_obj = y_val.shape[0]
                if y_val.ndim > 1:
                    raise ValueError(f"Function returned array with {y_val.ndim} dimensions. Expected 1 dimension for objectives (shape (M,)).")

            # Type and shape check for subsequent evaluations
            if y_val.shape[0] != self.n_obj:
                raise ValueError(f"Objective function returned {y_val.shape[0]} objectives, but expected {self.n_obj}.")

            y_evaluated_list.append(y_val)
            self.n_evals += 1

        if not y_evaluated_list:  # If no evaluations were made due to budget
            return np.array([])

        return np.array(y_evaluated_list)

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        # Update the archive with new evaluations.
        # Keep ALL evaluated points in the archive (self.X, self.y) for surrogate model training.
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def _is_pareto_efficient(self, costs: np.ndarray) -> np.ndarray:
        """
        Find the Pareto-efficient points. Assumes minimization for all objectives.
        :param costs: An (N, M) array of costs.
        :return: A boolean array of shape (N,). True for Pareto-efficient points.
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i in range(costs.shape[0]):
            if is_efficient[i]:  # Only check if not already marked as dominated
                # Check if costs[i] is dominated by any other point
                # A point costs[j] dominates costs[i] if:
                # all objectives of costs[j] are less than or equal to costs[i] (costs[j] <= costs[i])
                # AND at least one objective of costs[j] is strictly less than costs[i] (costs[j] < costs[i])
                dominated_by_others = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
                if np.any(dominated_by_others):
                    is_efficient[i] = False
        return is_efficient


    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Main optimization loop.
        # Always use _evaluate_points to call func so that the budget is respected.

        # 1. Initial Design
        initial_X = self._sample_points(self.n_init)
        new_y = self._evaluate_points(func, initial_X)
        # Only add points that were actually evaluated (e.g., if budget was very small)
        self._update_eval_points(initial_X[:new_y.shape[0]], new_y)

        # Optimization loop
        while self.n_evals < self.budget:
            # Check if there's enough data to fit a model (at least self.dim + 1 points for GP)
            min_data_for_model = max(self.dim + 1, 2)
            if self.X.shape[0] < min_data_for_model:
                # If not enough data, sample randomly to reach min_data_for_model or exhaust budget.
                points_to_sample = min(self.batch_size, self.budget - self.n_evals, min_data_for_model - self.X.shape[0])
                if points_to_sample <= 0:  # Safety check if budget very small or already met
                    break
                next_X_batch = self._sample_points(points_to_sample)
            else:
                # 2. Fit Surrogate Model
                self._fit_model(self.X, self.y)

                # 3. Select Next Points using Acquisition Function
                next_X_batch = self._select_next_points(min(self.batch_size, self.budget - self.n_evals))

            # 4. Evaluate Selected Points
            new_y_batch = self._evaluate_points(func, next_X_batch)

            # If no points were evaluated due to budget exhaustion or next_X_batch was empty, break
            if new_y_batch.size == 0:
                break

            # 5. Update Archive
            # Only update with points that were actually evaluated
            self._update_eval_points(next_X_batch[:new_y_batch.shape[0]], new_y_batch)

        # After loop, extract and return the final non-dominated set (Pareto front)
        if self.X is None or self.y is None or self.X.shape[0] == 0:
            # Return empty arrays if no evaluations were made (e.g., budget=0)
            return np.array([]), np.array([])

        pareto_indices = self._is_pareto_efficient(self.y)
        F_pareto = self.y[pareto_indices]
        X_pareto = self.X[pareto_indices]

        return F_pareto, X_pareto

