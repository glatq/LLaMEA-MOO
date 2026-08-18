from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors


class MOBORandomForestLCBPBI_CD:
    """
    A multi-objective Bayesian Optimization (MOBO) algorithm using Random Forest surrogate models
    and a hybrid acquisition function combining Lower Confidence Bound (LCB)-adjusted Penalty-Based Boundary Intersection (PBI)
    scalarization with a Crowding Distance-based term for diversity.
    It emphasizes scale-invariance through dynamic objective normalization and generalizability
    across varying dimensions and number of objectives.
    """

    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None,
                 initial_design_points: int = -1, # Default will be set dynamically
                 model_window_size: int = 150,
                 n_candidates_acquisition: int = 200,
                 n_weights_acquisition: int = 15,
                 rf_n_estimators: int = 20,
                 kappa_ucb: float = 0.1,
                 pbi_theta: float = 5.0, # PBI penalty parameter, now tunable
                 kappa_cd: float = 0.1):
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
        # Dynamic default for initial_design_points based on budget and dim
        if initial_design_points == -1: # sentinel value
            self.initial_design_points = min(max(5, 2 * self.dim + 1), self.budget // 4)
            if self.budget < 5: # For very small budgets, ensure at least one point
                self.initial_design_points = max(1, self.initial_design_points)
        else:
            self.initial_design_points = initial_design_points

        self.model_window_size = model_window_size
        self.n_candidates_acquisition = n_candidates_acquisition
        self.n_weights_acquisition = n_weights_acquisition
        self.rf_n_estimators = rf_n_estimators
        self.kappa_ucb = kappa_ucb # UCB exploration coefficient
        self.pbi_theta = pbi_theta # PBI penalty parameter, now tunable
        self.kappa_cd = kappa_cd # Crowding distance exploration coefficient
        self.rf_max_features = "sqrt" # Fixed for simplicity and robustness, common good default for regression

        # The number of objectives (self.n_obj) is unknown a priori.
        # It MUST be inferred on the first call to func inside _evaluate_points.
        self.n_obj: int | None = None

        # X has shape (n_points, n_dims), y has shape (n_points, n_obj)
        self.X: np.ndarray | None = None  # Archive of all evaluated design points
        self.y: np.ndarray | None = None  # Archive of all evaluated objective values
        self.n_evals = 0  # the number of function evaluations

        # The actual number of initial evaluations based on budget
        self.n_init = min(self.initial_design_points, max(1, self.budget))

        # Internal state for the surrogate model
        self.rf_models: list[RandomForestRegressor] = [] # List of RandomForestRegressor models, one per objective

        # Internal state for Pareto front (X_pareto and F_pareto will be updated incrementally)
        self.F_pareto: np.ndarray | None = None
        self.X_pareto: np.ndarray | None = None

    def _sample_points(self, n_points: int) -> np.ndarray:
        # Sample n_points candidate points efficiently within self.bounds.
        # Use self.bounds[0] as lower bounds and self.bounds[1] as upper bounds.
        # Return array of shape (n_points, n_dims).
        # Use a consistent seed for reproducibility but vary it for different runs of SMAC.
        # np.random.randint(0, 2**30) ensures a new seed each time this method is called within a run,
        # but the overall run might be seeded externally by SMAC.
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=np.random.randint(0, 2**30))
        samples_unit_cube = sampler.random(n_points)
        # Scale samples to the actual bounds
        scaled_samples = qmc.scale(samples_unit_cube, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X_all: np.ndarray, y_all: np.ndarray):
        # Fit a surrogate model on (X_all, y_all).
        # Apply sliding window: use only the most recent points
        if X_all.shape[0] > self.model_window_size:
            X_train = X_all[-self.model_window_size:]
            y_train = y_all[-self.model_window_size:]
        else:
            X_train = X_all
            y_train = y_all
        
        self.rf_models = []
        for i in range(self.n_obj): # Iterate over objectives
            rf = RandomForestRegressor(n_estimators=self.rf_n_estimators,
                                       max_features=self.rf_max_features,
                                       random_state=42, # Fixed random state for internal RF reproducibility
                                       n_jobs=-1) # Use all available cores
            rf.fit(X_train, y_train[:, i])
            self.rf_models.append(rf)

    def _get_predictions_with_std(self, X_cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Predict mean and standard deviation for each objective for candidate points.
        mu_preds = np.zeros((X_cand.shape[0], self.n_obj))
        sigma_preds = np.zeros((X_cand.shape[0], self.n_obj))

        for i, rf_model in enumerate(self.rf_models):
            # Predict mean
            mu_preds[:, i] = rf_model.predict(X_cand)

            # Predict std from individual tree predictions (heuristic for RandomForest)
            # This involves getting predictions from each tree and computing their standard deviation.
            # all_tree_preds_for_obj_i will be (n_estimators, n_candidates)
            all_tree_preds_for_obj_i = np.array([tree.predict(X_cand) for tree in rf_model.estimators_])
            sigma_preds[:, i] = np.std(all_tree_preds_for_obj_i, axis=0)
            # Add a small epsilon to std to prevent division by zero or numerical instability in acquisition
            sigma_preds[:, i] += 1e-8 

        return mu_preds, sigma_preds

    def _acquisition_function(self, X_cand: np.ndarray) -> np.ndarray:
        # Implement a multi-objective acquisition function: PBI-LCB with Crowding Distance.

        # If not enough data to fit model or normalize, return random scores for initial exploration
        # A minimum of (n_obj + 1) points is a common heuristic for model stability
        if self.X is None or self.y is None or self.y.shape[0] < max(2, self.n_obj + 1):
            return np.random.rand(X_cand.shape[0])

        # 1. Predict mean and standard deviation for candidate points
        mu_preds, sigma_preds = self._get_predictions_with_std(X_cand)

        # 2. Dynamically normalize objectives based on current observed data
        y_min_obs = np.min(self.y, axis=0) # Estimated Ideal Point (assuming minimization)
        y_max_obs = np.max(self.y, axis=0) # Estimated Nadir Point (assuming minimization)
        y_range = y_max_obs - y_min_obs
        # Add a small epsilon to avoid division by zero if all observed values for an objective are identical
        y_range[y_range == 0] = 1e-6

        # Normalize predictions
        mu_norm = (mu_preds - y_min_obs) / y_range
        sigma_norm = sigma_preds / y_range

        # Normalize current Pareto front for crowding distance calculation
        # Handle case where F_pareto is None or empty.
        if self.F_pareto is None or self.F_pareto.shape[0] == 0:
            F_pareto_norm = np.array([]).reshape(0, self.n_obj)
        else:
            F_pareto_norm = (self.F_pareto - y_min_obs) / y_range
        
        # 3. Generate diverse weight vectors (for PBI scalarization)
        # Sobol sequence for points on a simplex (weights sum to 1)
        sampler = qmc.Sobol(d=self.n_obj, scramble=True, seed=np.random.randint(0, 2**30))
        weights_raw = sampler.random(self.n_weights_acquisition)
        # Normalize weights to sum to 1
        weights = weights_raw / np.sum(weights_raw, axis=1, keepdims=True)
        # Ensure no zero weights to avoid issues with PBI, add a small epsilon
        weights[weights == 0] = 1e-6
        weights = weights / np.sum(weights, axis=1, keepdims=True) # Re-normalize

        # 4. Calculate PBI-LCB acquisition for each candidate
        # For minimization, we apply the UCB term to reduce the objective value (make it more "optimistic")
        obj_lcb_norm = mu_norm - self.kappa_ucb * sigma_norm
        
        acquisition_scores_per_weight = np.zeros((X_cand.shape[0], self.n_weights_acquisition))
        
        for w_idx, w in enumerate(weights):
            w_norm = w / np.linalg.norm(w) # Normalized weight vector for projection

            # PBI for LCB-adjusted objectives (f = obj_lcb_norm)
            d1_lcb = np.sum(w * obj_lcb_norm, axis=1) / np.linalg.norm(w)
            proj_lcb = d1_lcb[:, np.newaxis] * w_norm
            d2_lcb = np.linalg.norm(obj_lcb_norm - proj_lcb, axis=1)
            pbi_lcb = d1_lcb + self.pbi_theta * d2_lcb
            
            acquisition_scores_per_weight[:, w_idx] = pbi_lcb

        # Combine scores across weight vectors: take the minimum (most optimistic estimate)
        min_pbi_lcb_scores = np.min(acquisition_scores_per_weight, axis=1)

        # 5. Calculate Crowding Distance (CD) proxy for diversity
        # This term should decrease the acquisition score if the candidate is in a sparse region (making it more attractive).
        crowding_term = np.zeros(X_cand.shape[0])
        
        # Only calculate crowding distance if there are existing Pareto points to compare against
        if F_pareto_norm.shape[0] > 0:
            # Fix k_density_neighbors to a small value (e.g., 3) for stability and speed
            k_actual = min(3, F_pareto_norm.shape[0]) 
            if k_actual < 1:
                k_actual = 1 # Ensure at least 1 neighbor for density calculation
            
            # Use NearestNeighbors to find distances in objective space
            # n_jobs=-1 for parallel computation if possible
            nn_obj_space = NearestNeighbors(n_neighbors=k_actual, n_jobs=-1)
            nn_obj_space.fit(F_pareto_norm)
            
            # Get distances to k-nearest neighbors in objective space for each candidate's predicted mean
            distances_to_pareto, _ = nn_obj_space.kneighbors(mu_norm)
            
            # Take the average distance to the k-neighbors as a density proxy.
            # Higher average distance means sparser region, which is desirable for diversity.
            crowding_term = np.mean(distances_to_pareto, axis=1)

        # Final acquisition score: PBI-LCB (minimize) - kappa_cd * crowding_term (maximize crowding_term)
        # A large crowding_term (sparse region) makes the acquisition score lower (more attractive).
        acquisition_scores = min_pbi_lcb_scores - self.kappa_cd * crowding_term

        return acquisition_scores

    def _select_next_points(self, batch_size: int) -> np.ndarray:
        # Select the next points to evaluate using the acquisition function.
        # Generate candidate points and score them with the acquisition function.
        # Return an array of shape (batch_size, n_dims).

        # Generate candidate points for acquisition
        candidate_X = self._sample_points(self.n_candidates_acquisition)

        # If model is not yet fitted (e.g., during initial design phase or not enough data), sample randomly
        # A minimum of (n_obj + 1) points is a common heuristic for model stability.
        if not self.rf_models or self.X.shape[0] < max(2, self.n_obj + 1):
            return self._sample_points(batch_size)

        # Calculate acquisition scores
        acquisition_scores = self._acquisition_function(candidate_X)

        # Select the point(s) with the lowest acquisition score (assuming minimization)
        sorted_indices = np.argsort(acquisition_scores)
        next_X = candidate_X[sorted_indices[:batch_size]]
        
        return next_X

    def _evaluate_points(self, func: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> np.ndarray:
        # Evaluate the points in X.
        # On the first evaluation, infer M from func(x). Set self.n_obj = y.shape[0] and enforce this dimension in all later operations.
        # This method must be the only place where func is called.
        # Respect the remaining budget: do not exceed self.budget evaluations in total.
        # Use simple, clear looping over points, and clip points to the bounds if necessary.
        # Update self.n_evals by the actual number of function calls performed.
        # Clip points to self.bounds before calling func, to respect the problem domain.
        
        if self.n_evals >= self.budget:
            # If no objectives have been inferred yet, return an empty array with 0 columns.
            # Otherwise, return an empty array with self.n_obj columns.
            return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0)

        num_to_evaluate = min(len(X), self.budget - self.n_evals)
        if num_to_evaluate == 0:
            return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0)

        X_to_eval = X[:num_to_evaluate]
        X_clipped = np.clip(X_to_eval, self.bounds[0], self.bounds[1])

        y_evaluated_list = []
        for x_c in X_clipped:
            y_val = func(x_c)
            if self.n_obj is None:
                self.n_obj = y_val.shape[0]
            elif y_val.shape[0] != self.n_obj:
                raise ValueError(f"Expected {self.n_obj} objectives, but func returned {y_val.shape[0]}.")
            
            y_evaluated_list.append(y_val)
            self.n_evals += 1
            if self.n_evals >= self.budget:
                break # Stop if budget is exhausted mid-batch

        return np.array(y_evaluated_list)

    def _update_pareto_front_incremental(self, new_X_point: np.ndarray, new_y_point: np.ndarray):
        # Update self.F_pareto and self.X_pareto with a new point incrementally.
        # Assumes minimization for all objectives.

        # If no Pareto front exists yet, initialize it with the first point.
        if self.F_pareto is None or self.F_pareto.shape[0] == 0:
            self.F_pareto = np.array([new_y_point])
            self.X_pareto = np.array([new_X_point])
            return

        # Check if new_y_point is dominated by any existing Pareto point
        is_dominated_by_existing = False
        # Create a boolean array for efficient comparison
        existing_dominates = np.all(self.F_pareto <= new_y_point, axis=1) & np.any(self.F_pareto < new_y_point, axis=1)
        if np.any(existing_dominates):
            is_dominated_by_existing = True
        
        if is_dominated_by_existing:
            return # New point is dominated, not part of Pareto front

        # Identify points from current F_pareto that are now dominated by new_y_point
        # new_y_point dominates p_y if new_y_point is better than or equal in all objectives
        # AND strictly better in at least one.
        new_point_dominates = np.all(new_y_point <= self.F_pareto, axis=1) & np.any(new_y_point < self.F_pareto, axis=1)
        
        # Keep points that are NOT dominated by new_y_point
        self.F_pareto = self.F_pareto[~new_point_dominates]
        self.X_pareto = self.X_pareto[~new_point_dominates]
        
        # Add the new non-dominated point
        self.F_pareto = np.vstack((self.F_pareto, new_y_point))
        self.X_pareto = np.vstack((self.X_pareto, new_X_point))

    def _update_eval_points(self, new_X_batch: np.ndarray, new_y_batch: np.ndarray):
        # Update the archive with new evaluations.
        # Keep ALL evaluated points in the archive (self.X, self.y) for surrogate model training.
        # Track the non-dominated (Pareto) front separately for returning results.

        if self.X is None:
            self.X = new_X_batch
            self.y = new_y_batch
        else:
            self.X = np.vstack((self.X, new_X_batch))
            self.y = np.vstack((self.y, new_y_batch))

        # Update Pareto front incrementally for each new point in the batch
        for i in range(new_X_batch.shape[0]):
            self._update_pareto_front_incremental(new_X_batch[i], new_y_batch[i])


    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Main optimization loop.
        # func takes an array of shape (n_dims,) and returns np.ndarray of shape (M,) (one value per objective).
        # Always use _evaluate_points to call func so that the budget is respected.
        # The algorithm MUST remain correct for any number of objectives self.n_obj >= 2 without code changes.

        # Initial Design Phase
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        
        # If budget was 0 or initial_y is empty for some reason, return empty arrays.
        if initial_y.shape[0] == 0:
            return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0), \
                   np.array([]).reshape(0, self.dim)

        self._update_eval_points(initial_X, initial_y)

        # Main Optimization Loop (Surrogate-assisted)
        while self.n_evals < self.budget:
            # Fit surrogate model on current archive
            # Ensure enough points for training, otherwise skip model fitting and sample randomly
            # A minimum of (n_obj + 1) points is a common heuristic for model stability.
            if self.X.shape[0] < max(2, self.n_obj + 1): 
                 next_X = self._sample_points(1)
            else:
                self._fit_model(self.X, self.y)
                # Select next point(s) using acquisition function
                next_X = self._select_next_points(batch_size=1) # Evaluate one point at a time for stability

            # Evaluate the selected point(s)
            new_y = self._evaluate_points(func, next_X)

            # If budget ran out during evaluation, new_y might be empty
            if new_y.shape[0] == 0:
                break

            # Update archive and Pareto front
            self._update_eval_points(next_X, new_y)

        # Return the final non-dominated set (F_pareto) and corresponding design points (X_pareto)
        # Ensure F_pareto and X_pareto are not None and have correct dimensions even if empty
        if self.F_pareto is None:
            return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0), \
                   np.array([]).reshape(0, self.dim)
        return self.F_pareto, self.X_pareto
