from collections.abc import Callable
from scipy.stats import qmc, dirichlet
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class MOBORandomForestParEGO_Batch_Refined:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None,
                 n_candidates: int = 200,  # Number of candidate points for acquisition optimization
                 sliding_window_size: int = 100,  # Max number of recent points for model training
                 batch_size: int = 1,  # Number of points to evaluate in each iteration
                 n_estimators: int = 50,  # Number of trees in the Random Forest
                 max_features: str | float = "sqrt",  # Max features for Random Forest (e.g., "sqrt", "log2", 0.5)
                 min_samples_leaf: int = 1,  # Minimum samples required to be at a leaf node in RF
                 exploration_beta: float = 2.0  # Parameter for LCB-style exploration in acquisition
                 ):
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
        self.n_candidates = n_candidates
        self.sliding_window_size = sliding_window_size
        self.batch_size = batch_size
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.exploration_beta = exploration_beta

        # The number of objectives (self.n_obj) is unknown a priori.
        # It MUST be inferred on the first call to func inside _evaluate_points.
        self.n_obj: int | None = None

        # X has shape (n_points, n_dims), y has shape (n_points, n_obj)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.n_evals = 0  # the number of function evaluations

        # Choose a reasonable number of initial evaluations.
        # This parameter is not part of the SMAC configuration space, as it depends on problem specifics (dim, budget).
        # Ensures enough points for initial modeling without consuming too much budget.
        self.n_init = max(2 * dim + 1, min(10, self.budget // 4)) 
        # Ensure n_init does not exceed budget.
        self.n_init = min(self.n_init, self.budget)
        
        # Random Forest Regressors for each objective
        self.rf_models: list[RandomForestRegressor] = []

    def _sample_points(self, n_points: int) -> np.ndarray:
        # Sample n_points candidate points efficiently within self.bounds.
        # Use self.bounds[0] as lower bounds and self.bounds[1] as upper bounds.
        # Return array of shape (n_points, n_dims).
        
        # Use Latin Hypercube Sampling (LHS) for efficient space-filling.
        # Add seed for reproducibility within a run.
        # Use a new random seed for each call to ensure variety if called multiple times.
        sampler = qmc.LatinHypercube(d=self.dim, seed=np.random.randint(0, 100000)) 
        samples = sampler.random(n=n_points)
        
        # Scale samples to the actual bounds
        lb = self.bounds[0]
        ub = self.bounds[1]
        scaled_samples = lb + samples * (ub - lb)
        
        return scaled_samples

    def _get_sliding_window_data(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Retrieves data for model training using a sliding window.
        """
        if self.X is None or len(self.X) <= self.sliding_window_size:
            return self.X, self.y
        
        # Use the most recent 'sliding_window_size' points
        return self.X[-self.sliding_window_size:], self.y[-self.sliding_window_size:]

    def _fit_model(self):
        # Fit a surrogate model on (X, y).
        # Return the fitted model and store any required state on self.
        # Do not change the function signature.
        # Consider using a sliding window (e.g., most recent 200-500 points) to keep fitting tractable.
        
        X_train, y_train = self._get_sliding_window_data()

        if X_train is None or len(X_train) == 0:
            raise ValueError("Cannot fit model with no training data.")
        
        # Initialize RF models if not already done, or if n_obj changed (shouldn't happen after first eval)
        if not self.rf_models or len(self.rf_models) != self.n_obj:
            self.rf_models = [
                RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_features=self.max_features,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=42, # For reproducibility of RF internal state
                    n_jobs=-1 # Use all available cores
                ) for _ in range(self.n_obj)
            ]

        # Fit each RF model for each objective
        for i in range(self.n_obj):
            self.rf_models[i].fit(X_train, y_train[:, i])

    def _predict_with_uncertainty(self, X_predict: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and empirical standard deviation for each objective using the Random Forest ensemble.
        The standard deviation is calculated from the predictions of individual trees.
        """
        mu_all_obj = np.zeros((len(X_predict), self.n_obj))
        sigma_all_obj = np.zeros((len(X_predict), self.n_obj))

        for i in range(self.n_obj):
            # Get predictions from each tree in the forest
            # RandomForestRegressor.estimators_ already contains fitted trees.
            tree_predictions = np.array([tree.predict(X_predict) for tree in self.rf_models[i].estimators_])
            
            # Mean prediction is the mean of tree predictions
            mu_all_obj[:, i] = np.mean(tree_predictions, axis=0)
            
            # Empirical standard deviation is the std of tree predictions
            sigma_all_obj[:, i] = np.std(tree_predictions, axis=0)
            
            # Ensure sigma is non-negative and has a minimum value to avoid numerical instability
            sigma_all_obj[:, i] = np.maximum(1e-9, sigma_all_obj[:, i])
            
        return mu_all_obj, sigma_all_obj

    def _select_next_points(self) -> np.ndarray:
        # Select the next points to evaluate using the ParEGO-inspired acquisition strategy.
        # Generates 'batch_size' unique points, each selected by optimizing the acquisition
        # function with a new, randomly sampled weight vector.
        # Returns an array of shape (batch_size, n_dims).
        
        next_X_batch = []
        all_candidates = self._sample_points(self.n_candidates)

        # Pre-calculate predictions for all candidates once
        mu_all_obj_candidates, sigma_all_obj_candidates = self._predict_with_uncertainty(all_candidates)

        # Pre-calculate observed min/max for normalization
        # Assumes objectives are to be minimized.
        y_min_observed = np.min(self.y, axis=0)
        y_max_observed = np.max(self.y, axis=0)
        y_range = y_max_observed - y_min_observed
        y_range[y_range == 0] = 1e-6 # Avoid division by zero

        # Normalize predicted means and standard deviations using the same range
        # Note: If y_min_observed is large, mu_all_obj_candidates - y_min_observed can be negative.
        # This is fine as we are minimizing.
        normalized_mu_candidates = (mu_all_obj_candidates - y_min_observed) / y_range
        normalized_sigma_candidates = sigma_all_obj_candidates / y_range

        # Keep track of which candidates have been selected in this batch
        # Using a boolean mask for efficiency
        is_candidate_selected = np.zeros(len(all_candidates), dtype=bool)

        for _ in range(self.batch_size):
            # If all candidates are selected or budget is exhausted, break
            if np.all(is_candidate_selected) or self.n_evals + len(next_X_batch) >= self.budget:
                break

            # Generate a new random weight vector for this selection in the batch
            # Dirichlet distribution (alpha=1 for uniform sampling over the simplex)
            # Use a new random seed for each call to ensure variety.
            weights = dirichlet.rvs(np.ones(self.n_obj), random_state=np.random.randint(0, 100000))[0] 

            # Calculate acquisition scores for ALL candidates
            # Tchebycheff scalarization for mean: max_j(w_j * normalized_mu_j)
            # Tchebycheff scalarization for std (heuristic): max_j(w_j * normalized_sigma_j)
            weighted_normalized_mu = weights * normalized_mu_candidates
            weighted_normalized_sigma = weights * normalized_sigma_candidates

            t_mu_scalarized = np.max(weighted_normalized_mu, axis=1)
            t_sigma_scalarized = np.max(weighted_normalized_sigma, axis=1)

            # Acquisition function: -(Tchebycheff_mean - exploration_beta * Tchebycheff_sigma)
            # Maximizing this means minimizing the weighted Tchebycheff value while exploring uncertain regions.
            acquisition_scores = - (t_mu_scalarized - self.exploration_beta * t_sigma_scalarized)

            # Mask out already selected candidates by assigning them a very low score
            acquisition_scores[is_candidate_selected] = -np.inf 

            # Find the best candidate from the remaining ones
            best_idx = np.argmax(acquisition_scores)
            
            # Fallback if no valid unselected candidates remain (e.g., all have -np.inf score)
            if acquisition_scores[best_idx] == -np.inf:
                # If no unselected candidate has a finite score, this implies an issue or all candidates are equally bad.
                # In this case, just pick a random unselected candidate if available.
                unselected_indices = np.where(~is_candidate_selected)[0]
                if len(unselected_indices) > 0:
                    best_idx = np.random.choice(unselected_indices)
                else: 
                    # No unselected candidates left, break the batch selection loop.
                    break 

            next_X_batch.append(all_candidates[best_idx])
            is_candidate_selected[best_idx] = True # Mark this candidate as selected
            
        return np.array(next_X_batch)

    def _evaluate_points(self, func: Callable[[np.ndarray], np.ndarray], X_eval: np.ndarray) -> np.ndarray:
        # Evaluate the points in X_eval.
        # On the first evaluation, infer M from func(x). Set self.n_obj = y.shape[0] and enforce this dimension in all later operations.
        # This method must be the only place where func is called.
        # Respect the remaining budget: do not exceed self.budget evaluations in total.
        # Clip points to self.bounds before calling func, to respect the problem domain.
        # Update self.n_evals by the actual number of function calls performed.
        
        if self.n_evals >= self.budget:
            return np.array([]) # No budget left

        # Determine how many points can actually be evaluated within the budget
        points_to_evaluate = min(len(X_eval), self.budget - self.n_evals)
        if points_to_evaluate == 0:
            return np.array([])

        # Clip points to bounds before evaluation
        X_clipped = np.zeros_like(X_eval[:points_to_evaluate])
        lb, ub = self.bounds[0], self.bounds[1]
        for i in range(points_to_evaluate):
            X_clipped[i] = np.clip(X_eval[i], lb, ub)

        y_results = []
        for i in range(points_to_evaluate):
            y_val = func(X_clipped[i])
            
            # Infer n_obj on the first evaluation
            if self.n_obj is None:
                self.n_obj = len(y_val)
            # Ensure consistent number of objectives
            elif len(y_val) != self.n_obj:
                raise ValueError(f"Objective function returned {len(y_val)} objectives, but expected {self.n_obj}.")
            
            y_results.append(y_val)
            self.n_evals += 1
            if self.n_evals >= self.budget:
                break # Stop if budget is exhausted mid-batch

        # Convert list of results to a NumPy array. Handle empty list case.
        if y_results:
            return np.array(y_results)
        else:
            return np.array([])

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        # Update the archive with new evaluations.
        # Do not change the function signature.
        # IMPORTANT: Keep ALL evaluated points in the archive (self.X, self.y) for surrogate model training.
        # The surrogate model needs both dominated and non-dominated points to learn the objective landscape.
        # Track the non-dominated (Pareto) front separately for returning results.
        # Dominance comparisons MUST use y.shape[1] == self.n_obj, never a fixed number of objectives.
        
        if self.X is None or len(self.X) == 0:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def _get_pareto_front(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts the non-dominated (Pareto) front from the current archive.
        Assumes minimization for all objectives.
        """
        if self.y is None or len(self.y) == 0:
            # If n_obj is known, return empty arrays with correct shape for F, otherwise 0 for F's dim.
            if self.n_obj is not None:
                return np.array([]).reshape(0, self.n_obj), np.array([]).reshape(0, self.dim)
            else:
                return np.array([]).reshape(0, 0), np.array([]).reshape(0, self.dim)


        is_dominated = np.zeros(len(self.y), dtype=bool)
        for i in range(len(self.y)):
            for j in range(len(self.y)):
                if i == j:
                    continue
                # Check if y[j] dominates y[i]
                # y[j] dominates y[i] if y[j] <= y[i] for all objectives AND y[j] < y[i] for at least one objective
                if np.all(self.y[j] <= self.y[i]) and np.any(self.y[j] < self.y[i]):
                    is_dominated[i] = True
                    break
        
        pareto_X = self.X[~is_dominated]
        pareto_F = self.y[~is_dominated]
        
        return pareto_F, pareto_X

    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Main optimization loop.
        # func takes an array of shape (n_dims,) and returns np.ndarray of shape (M,) (one value per objective).
        # Always use _evaluate_points to call func so that the budget is respected.
        # The full run must complete within approximately 60 seconds.
        # Return a tuple (F_pareto, X_pareto), where F_pareto has shape (K, n_obj)
        # and X_pareto has shape (K, n_dims) for the final non-dominated set.
        # The algorithm MUST remain correct for any number of objectives self.n_obj >= 2 without code changes.

        # 1. Initial Design Phase: Evaluate initial points to populate archive and infer n_obj.
        # This ensures we have a baseline and n_obj is set before the main loop.
        
        # Evaluate up to self.n_init points (or fewer if budget is smaller)
        initial_points_to_eval = self.n_init
        
        if initial_points_to_eval > 0:
            initial_X = self._sample_points(initial_points_to_eval)
            initial_y = self._evaluate_points(func, initial_X)
            if initial_y.size > 0:
                self._update_eval_points(initial_X[:len(initial_y)], initial_y)
            
            # If after initial evaluation, n_obj is still None (e.g., budget was too low to get any eval)
            # or if no points were evaluated, then we cannot proceed.
            if self.n_obj is None or self.y is None or len(self.y) == 0:
                # Return empty Pareto front if no evaluations could be made or n_obj not determined
                return (np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0), 
                        np.array([]).reshape(0, self.dim))

        # Main optimization loop
        while self.n_evals < self.budget:
            # Ensure enough data for model training before fitting.
            # A minimum number of points is needed for RF to be effective.
            # Use a heuristic: at least 2*dim + 1, or 2*n_obj, or a fixed minimum like 10, whichever is largest.
            required_min_points_for_model = max(2 * self.dim + 1, 2 * self.n_obj, 10)
            
            if len(self.X) < required_min_points_for_model:
                n_additional_needed = required_min_points_for_model - len(self.X)
                n_to_sample = min(n_additional_needed, self.budget - self.n_evals)
                
                if n_to_sample <= 0: # No budget left or already enough points
                    break

                additional_X = self._sample_points(n_to_sample)
                additional_y = self._evaluate_points(func, additional_X)
                if additional_y.size > 0:
                    self._update_eval_points(additional_X[:len(additional_y)], additional_y)
                
                # Check if budget exhausted after evaluating additional points
                if self.n_evals >= self.budget:
                    break
                # If still not enough points after trying to get more, something is wrong or budget very tight.
                # This check prevents potential infinite loops if model requirements can never be met.
                if len(self.X) < required_min_points_for_model:
                    break

            # 2. Fit Surrogate Models
            self._fit_model()

            # 3. Select Next Points using Acquisition Function (ParEGO-style batching)
            next_X = self._select_next_points()

            # 4. Evaluate Points
            next_y = self._evaluate_points(func, next_X)
            
            # 5. Update Archive
            if next_y.size > 0: # Only update if new points were actually evaluated
                self._update_eval_points(next_X[:len(next_y)], next_y)
            
        # 6. Extract Pareto Front
        F_pareto, X_pareto = self._get_pareto_front()

        return F_pareto, X_pareto

