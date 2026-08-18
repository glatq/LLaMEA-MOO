import numpy as np
import logging
from collections.abc import Callable
from scipy.stats import qmc
from scipy.spatial.distance import cdist # For farthest point sampling

# External libraries for surrogate models
try:
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor
except ImportError:
    logging.warning("scikit-learn not found. BaggingRegressor and DecisionTreeRegressor will not be available. "
                    "Please install scikit-learn for full functionality (pip install scikit-learn).")
    # Define dummy classes if sklearn is not available, to allow code structure to pass.
    # This prevents import errors but will result in non-functional models.
    class DecisionTreeRegressor:
        def __init__(self, *args, **kwargs): pass
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(X.shape[0])

    class BaggingRegressor:
        def __init__(self, base_estimator=None, n_estimators=10, *args, **kwargs):
            self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeRegressor()
            self.n_estimators = n_estimators
            self.estimators_ = [self.base_estimator for _ in range(self.n_estimators)] # Dummy estimators
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(X.shape[0])


# Helper function for non-dominated sorting (assumes maximization)
def is_pareto_efficient(costs: np.ndarray, return_indices: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Find the Pareto-efficient points in a set of N points in M dimensions.
    Assumes maximization (higher values are better).
    :param costs: An (N, M) array of objective values.
    :param return_indices: If True, return the indices of the Pareto-efficient points.
    :return: A boolean array of shape (N,) indicating Pareto-efficient points,
             or (N,) boolean array and (K,) integer array if return_indices is True.
    """
    if costs.shape[0] == 0:
        if return_indices:
            return np.array([], dtype=bool), np.array([], dtype=int)
        return np.array([], dtype=bool)

    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:  # Only check if point 'i' is still considered efficient
            # Create a mask for other points (excluding point i itself)
            other_points_mask = np.ones(costs.shape[0], dtype=bool)
            other_points_mask[i] = False
            
            # Filter out points that are already marked as non-efficient to speed up
            # This is an important optimization for performance.
            other_points_mask = other_points_mask & is_efficient
            
            if np.any(other_points_mask): # Only proceed if there are other points to compare against
                # Check if 'c' (costs[i]) is dominated by any of the 'other_points'
                # A point 'j' dominates 'i' if all objectives of 'j' are greater than or equal to 'i'
                # AND at least one objective of 'j' is strictly greater than 'i'.
                if np.any((np.all(costs[other_points_mask] >= c, axis=1)) & (np.any(costs[other_points_mask] > c, axis=1))):
                    is_efficient[i] = False
    
    if return_indices:
        return is_efficient, np.where(is_efficient)[0]
    return is_efficient


# Helper to generate weights on the simplex
def _generate_weights_on_simplex(n_obj: int, n_weights: int) -> np.ndarray:
    """
    Generates n_weights uniformly distributed weight vectors on the simplex.
    Each vector sums to 1.
    """
    if n_obj == 1:
        return np.array([[1.0]])
    
    # Use Dirichlet distribution for uniform sampling on simplex
    # Alpha = [1, ..., 1] makes it uniform.
    weights = np.random.dirichlet(np.ones(n_obj), n_weights)
    return weights


class MOBORobustLCBFPS: # Multi-Objective Bayesian Optimization with Robust LCB and Farthest Point Sampling
    """
    A multi-objective Bayesian Optimization algorithm that combines the robustness of Bagging Regressors
    with a scalarized Lower Confidence Bound (LCB) acquisition function and a diversity-enhanced
    batch selection strategy using Farthest Point Sampling (FPS) in the predicted objective space.
    Objectives are dynamically normalized for scale-invariant performance.
    """
    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None,
                 n_candidates_per_batch: int = 300, # Hyperparameter for SMAC
                 model_window_size: int = 150, # Hyperparameter for SMAC
                 beta: float = 2.0, # Hyperparameter for SMAC
                 n_selection_pool: int = 60, # Hyperparameter for SMAC
                 bag_n_estimators: int = 25, # Hyperparameter for SMAC
                 bag_max_features_str: str = "sqrt", # Hyperparameter for SMAC
                 bag_min_samples_leaf: int = 5): # Hyperparameter for SMAC
        
        # Fixed problem parameters
        self.budget = budget
        self.dim = dim
        # bounds has shape (2, dim), bounds[0]: lower bound, bounds[1]: upper bound
        if bounds is None:
            # Fallback: assume a simple [0, 1]^dim box if no bounds are provided.
            self.bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        else:
            self.bounds = np.asarray(bounds, dtype=float)

        # Fixed parameters (not tuned by SMAC to keep config space compact)
        self.batch_size: int = 2 # Fixed: Number of points to evaluate in each iteration
        self.n_weights: int = 20 # Fixed: Number of weight vectors for scalarization

        # Calculate n_initial_design robustly and budget-aware, not a hyperparameter
        # min(budget // 4, 50) ensures it's not too large and leaves budget for BO iterations.
        # max(self.dim + 1, ...) ensures enough points for stable model training.
        self.n_initial_design = max(self.dim + 1, min(self.budget // 4, 2 * self.dim + 1, 50))
        # Ensure n_initial_design leaves room for at least one batch evaluation
        self.n_initial_design = min(self.n_initial_design, self.budget - self.batch_size)


        # Hyperparameters (tuned by SMAC, defined in Space)
        self.n_candidates_per_batch = n_candidates_per_batch
        self.model_window_size = min(budget, model_window_size) # Ensure window size doesn't exceed budget
        self.beta = beta # Exploration-exploitation trade-off for LCB: mu - beta * sigma (for minimization)
        self.n_selection_pool = min(n_selection_pool, self.n_candidates_per_batch) # Size of the pool for diversity selection
        self.bag_n_estimators = bag_n_estimators
        self.bag_max_features = self._parse_bag_max_features(bag_max_features_str)
        self.bag_min_samples_leaf = bag_min_samples_leaf

        # The number of objectives (self.n_obj) is unknown a priori.
        # It MUST be inferred on the first call to func inside _evaluate_points.
        self.n_obj: int | None = None

        # X has shape (n_points, n_dims), y has shape (n_points, n_obj)
        self.X: np.ndarray | None = None  # Original X points
        self.y: np.ndarray | None = None  # Original y values (to be maximized)
        self.n_evals = 0  # the number of function evaluations

        # Internal storage for negated objectives (for minimization-based acquisition)
        # y_internal = -y (assuming original y values are to be maximized)
        self.y_internal: np.ndarray | None = None
        self.y_min_internal: np.ndarray | None = None # Current observed ideal point (for negated objectives)
        self.y_max_internal: np.ndarray | None = None # Current observed nadir point (for negated objectives)

        # Bagging Regressor models, one per objective
        self.bagging_models: list[BaggingRegressor] = []

        # Epsilon for numerical stability
        self.epsilon = 1e-6

    def _parse_bag_max_features(self, bag_max_features_str: str):
        """Parses bag_max_features_str to the correct type for BaggingRegressor's base_estimator."""
        if bag_max_features_str in ["sqrt", "log2"]:
            return bag_max_features_str
        try:
            val = float(bag_max_features_str)
            if 0 < val <= 1: # Interpret as fraction if between 0 and 1
                return val
            else: # Interpret as integer number of features
                return int(val)
        except ValueError:
            logging.warning(f"Invalid bag_max_features_str: {bag_max_features_str}. Falling back to 'sqrt'.")
            return "sqrt"

    def _sample_points(self, n_points: int) -> np.ndarray:
        """
        Sample n_points candidate points efficiently within self.bounds using Latin Hypercube Sampling.
        Returns array of shape (n_points, n_dims).
        """
        if n_points <= 0:
            return np.array([]).reshape(0, self.dim)

        # Use a random seed for variability to avoid always sampling the same points across runs
        sampler = qmc.LatinHypercube(d=self.dim, seed=np.random.randint(0, 100000)) 
        # Sample in [0, 1]^dim and then scale to actual bounds
        sample = sampler.random(n=n_points)
        lb, ub = self.bounds[0], self.bounds[1]
        scaled_sample = qmc.scale(sample, lb, ub)
        return scaled_sample

    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit a surrogate model on (X_train, y_train).
        Trains self.n_obj independent Bagging Regressors.
        Uses a sliding window for training data to keep fitting tractable.
        """
        # BaggingRegressor needs at least min_samples_leaf samples to potentially split.
        # Also, need at least 2 samples if min_samples_leaf is 1 for some tree algorithms.
        min_samples_for_fit = max(2, self.bag_min_samples_leaf)
        if X_train.shape[0] < min_samples_for_fit:
            logging.debug(f"Not enough data ({X_train.shape[0]}) to train BaggingRegressor with min_samples_leaf={self.bag_min_samples_leaf}. Skipping model fit.")
            return

        # Select recent data for the sliding window
        if X_train.shape[0] > self.model_window_size:
            # Take the most recent 'model_window_size' points
            X_train = X_train[-self.model_window_size:]
            y_train = y_train[-self.model_window_size:]

        # Initialize Bagging Regressors if not already done, or if n_obj changed (shouldn't happen after first eval)
        if not self.bagging_models or len(self.bagging_models) != self.n_obj:
            self.bagging_models = [
                BaggingRegressor(base_estimator=DecisionTreeRegressor(
                                      max_features=self.bag_max_features,
                                      min_samples_leaf=self.bag_min_samples_leaf,
                                      random_state=i), # Different random state for each base tree
                                 n_estimators=self.bag_n_estimators,
                                 random_state=i, # Different random state for each Bagging model
                                 n_jobs=1) # Use 1 job to avoid multiprocessing issues and for predictability
                for i in range(self.n_obj)
            ]

        # Fit each Bagging Regressor for its respective objective
        for i in range(self.n_obj):
            try:
                self.bagging_models[i].fit(X_train, y_train[:, i])
            except Exception as e:
                logging.warning(f"BaggingRegressor fitting failed for objective {i}: {e}. Skipping update for this model.")

    def _predict_with_uncertainty(self, X_cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation for each objective using the Bagging Regressors.
        Uncertainty is estimated from the variance of predictions from individual trees.
        """
        if not self.bagging_models or X_cand.shape[0] == 0:
            # If models not fitted yet or no candidates, return dummy values.
            dummy_n_obj = self.n_obj if self.n_obj is not None else 2 # Assume 2 objectives for dummy if not known
            return np.zeros((X_cand.shape[0], dummy_n_obj)), np.ones((X_cand.shape[0], dummy_n_obj))

        mu_all_obj = np.zeros((X_cand.shape[0], self.n_obj))
        std_all_obj = np.zeros((X_cand.shape[0], self.n_obj))

        for i in range(self.n_obj):
            # Get predictions from individual trees in the BaggingRegressor
            # Note: estimators_ is a list of fitted base estimators.
            individual_tree_predictions = np.array([estimator.predict(X_cand) for estimator in self.bagging_models[i].estimators_])
            
            mu_all_obj[:, i] = np.mean(individual_tree_predictions, axis=0)
            std_all_obj[:, i] = np.std(individual_tree_predictions, axis=0)

        return mu_all_obj, std_all_obj

    def _acquisition_function(self, X_cand: np.ndarray) -> np.ndarray:
        """
        Calculates a scalarized Lower Confidence Bound (LCB) acquisition value for each candidate.
        Objectives are normalized based on current observed ranges, then LCBs are computed.
        Assumes minimization for internal calculations (objectives are negated internally).
        The acquisition function value is maximized.
        """
        # Check if enough data is available for meaningful model predictions and normalization.
        # It's crucial that self.y_internal contains enough points to establish a range.
        # self.n_initial_design is the minimum number of points evaluated before BO loop starts.
        if self.y_internal is None or self.y_internal.shape[0] < self.n_initial_design:
            # Not enough data to fit models or determine objective ranges reliably
            # Return uniform acquisition scores to encourage exploration
            return np.ones(X_cand.shape[0])

        mu_all_obj, std_all_obj = self._predict_with_uncertainty(X_cand)

        # Normalize predicted means and standard deviations based on current observed ranges
        # y_min_internal and y_max_internal refer to the min/max of the *negated* objectives
        # y_min_internal is the best (most negative), y_max_internal is the worst (least negative).
        range_internal = self.y_max_internal - self.y_min_internal
        
        # Handle cases where range is zero (all observed values are identical for an objective)
        # Set to 1.0 to avoid division by zero, effectively no scaling for that objective
        range_internal[range_internal < self.epsilon] = 1.0 

        # Normalize predicted means to [0,1] range (0=best, 1=worst for negated objectives)
        # This transforms mu values relative to the current observed Pareto front boundaries.
        mu_norm = (mu_all_obj - self.y_min_internal) / range_internal
        # Normalize predicted standard deviations.
        # This scales the uncertainty relative to the objective range.
        std_norm = std_all_obj / range_internal

        # Generate weight vectors for scalarization
        weights = _generate_weights_on_simplex(self.n_obj, self.n_weights)

        # Calculate scalarized LCB for each candidate under different weight vectors
        # For minimization, LCB = mu - beta * sigma. We want to find candidates that minimize this.
        # The acquisition function will be -LCB (to maximize it and find the best candidates).
        acquisition_scores = -np.inf * np.ones(X_cand.shape[0])

        # Iterate through weight vectors to cover different regions of the Pareto front
        for w in weights:
            # Scalarized LCB for this weight vector.
            # This combines the normalized mean and std for each objective.
            scalarized_lcb = np.sum(w * (mu_norm - self.beta * std_norm), axis=1)
            
            # We want to maximize the negative of scalarized_lcb across all weights
            # This is a common strategy to maximize expected improvement or hypervolume.
            acquisition_scores = np.maximum(acquisition_scores, -scalarized_lcb)
            
        # Ensure acquisition scores are non-negative, as they represent desirability.
        # Any score below zero would indicate a candidate worse than the current worst observed,
        # which isn't useful for selection.
        acquisition_scores[acquisition_scores < 0] = 0.0

        return acquisition_scores

    def _select_next_points(self, batch_size: int) -> np.ndarray:
        """
        Select the next points to evaluate using a diversity-enhanced strategy.
        1. Generate a large pool of candidate solutions via Latin Hypercube Sampling (LHS).
        2. Score these candidates using the scalarized LCB acquisition function.
        3. Form a "selection pool" by taking the top `n_selection_pool` candidates based on their acquisition scores.
        4. Apply Farthest Point Sampling (FPS) to this selection pool to choose `batch_size` points.
           FPS ensures diversity by selecting points that are maximally distant in the predicted
           mean objective space, while still being promising (high acquisition).
        """
        # Generate a large pool of candidate points for evaluation
        X_cand = self._sample_points(self.n_candidates_per_batch)
        
        if X_cand.shape[0] == 0:
            # If no candidates could be generated (e.g., n_candidates_per_batch=0 or dim=0),
            # return an empty array with correct shape.
            return np.array([]).reshape(0, self.dim)

        # Evaluate acquisition function for all candidates to quantify their promise
        acquisition_values = self._acquisition_function(X_cand)

        # Select the top 'n_selection_pool' candidates based on acquisition values.
        # This forms a smaller, high-quality pool from which diverse points will be chosen.
        # Robustness check: if all acquisition values are numerically identical, randomly select.
        if np.all(acquisition_values <= acquisition_values.min() + self.epsilon):
            logging.debug("All acquisition values are near identical. Reverting to random sampling for selection pool.")
            # Randomly select from all candidates if no clear best ones.
            top_indices = np.random.choice(X_cand.shape[0], size=min(self.n_selection_pool, X_cand.shape[0]), replace=False)
        else:
            # Sort in descending order and take the top N indices.
            sorted_indices = np.argsort(acquisition_values)[::-1]
            top_indices = sorted_indices[:min(self.n_selection_pool, X_cand.shape[0])]
        
        X_pool = X_cand[top_indices]
        pool_acquisition_values = acquisition_values[top_indices] # Acquisition values for points in the pool

        num_pool_points = X_pool.shape[0]
        if num_pool_points == 0: 
            # If the pool is empty after filtering, return empty array.
            return np.array([]).reshape(0, self.dim)
            
        # Ensure batch_size does not exceed available pool points.
        actual_batch_size = min(batch_size, num_pool_points)

        # Predict mean objectives for the pool. These are the *negated* objectives,
        # so convert them back to the original (maximization) scale for diversity sampling.
        mu_predicted_pool_internal, _ = self._predict_with_uncertainty(X_pool)
        mu_predicted_pool_original = -mu_predicted_pool_internal # Back to maximization scale for diversity sampling

        selected_indices_in_pool = []
        
        # 1. Select the point with the highest acquisition value from the pool as the first point for FPS.
        # This ensures that the most promising point is always included.
        first_idx_in_pool = np.argmax(pool_acquisition_values)
        selected_indices_in_pool.append(first_idx_in_pool)

        # If we only need 1 point, we are done.
        if actual_batch_size == 1:
            return X_pool[[first_idx_in_pool]]

        # 2. Iteratively select remaining points using Farthest Point Sampling (FPS).
        # FPS selects points that are maximally distant from already selected points,
        # promoting diverse exploration of the predicted Pareto front.
        for _ in range(actual_batch_size - 1):
            # Identify candidates that have not yet been selected.
            unselected_indices = np.setdiff1d(np.arange(num_pool_points), selected_indices_in_pool)
            
            if unselected_indices.shape[0] == 0:
                # All points in the pool are already selected, can happen if pool is small.
                break

            # Calculate the minimum Euclidean distance from each unselected point
            # to any of the currently selected points in the predicted objective space.
            distances = cdist(mu_predicted_pool_original[unselected_indices], 
                              mu_predicted_pool_original[selected_indices_in_pool], 
                              metric='euclidean')
            min_distances_to_selected = np.min(distances, axis=1)
            
            # Select the unselected point that has the maximum minimum distance.
            # This is the core logic of Farthest Point Sampling.
            next_idx_in_unselected_array = np.argmax(min_distances_to_selected)
            next_original_idx_in_pool = unselected_indices[next_idx_in_unselected_array]
            selected_indices_in_pool.append(next_original_idx_in_pool)

        return X_pool[selected_indices_in_pool]

    def _evaluate_points(self, func: Callable[[np.ndarray], np.ndarray], X_batch: np.ndarray) -> np.ndarray:
        """
        Evaluate the points in X_batch using the black-box function 'func'.
        This method is the only place where func is called.
        Infers self.n_obj on the first call.
        Updates self.n_evals.
        Clips points to self.bounds before calling func.
        Returns an array of shape (n_points, n_obj).
        """
        y_batch = []

        for x_point in X_batch:
            if self.n_evals >= self.budget:
                break # Stop if budget is exhausted
            
            # Clip point to bounds before calling func to ensure it's within the search space.
            x_clipped = np.clip(x_point, self.bounds[0], self.bounds[1])
            
            y_val = func(x_clipped)
            
            # Infer n_obj on the first evaluation. This is crucial for initializing models.
            if self.n_obj is None:
                self.n_obj = len(y_val)
                logging.info(f"Inferred number of objectives: {self.n_obj}")
                # Clear any potential dummy models that might have been created if sklearn was missing.
                self.bagging_models = []
                # Re-calculate n_initial_design if it was set to a default value that depends on n_obj (though not directly here)
                # Ensure n_initial_design is at least dim+1 for model training stability.
                self.n_initial_design = max(self.dim + 1, min(self.budget // 4, 2 * self.dim + 1, 50))
                self.n_initial_design = min(self.n_initial_design, self.budget - self.batch_size)


            # Ensure consistency of objective dimension for all subsequent evaluations.
            if len(y_val) != self.n_obj:
                raise ValueError(f"Function returned {len(y_val)} objectives, but {self.n_obj} was expected.")
            
            y_batch.append(y_val)
            self.n_evals += 1
        
        if not y_batch: # If no evaluations were made due to budget exhaustion or empty batch.
            # Return empty array with correct dimensions, handle case where n_obj might still be None.
            return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 1)
            
        return np.array(y_batch)

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        """
        Update the archive of evaluated points with new observations.
        Appends new_X and new_y to self.X and self.y respectively.
        Also updates internal negated objective storage (`self.y_internal`)
        and the current observed ideal/nadir points (`self.y_min_internal`, `self.y_max_internal`).
        """
        if new_X.shape[0] == 0: # No new points to add.
            return

        if self.X is None: # First points being added.
            self.X = new_X
            self.y = new_y
            self.y_internal = -new_y # Negate for internal minimization.
        else: # Append to existing archive.
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
            self.y_internal = np.vstack((self.y_internal, -new_y))

        # Update current observed min/max for internal (negated) objectives.
        # These are crucial for dynamic normalization of objective values.
        self.y_min_internal = np.min(self.y_internal, axis=0) # Ideal point (for negated objs).
        self.y_max_internal = np.max(self.y_internal, axis=0) # Nadir point (for negated objs).

    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Main optimization loop. Executes the Bayesian Optimization process.
        """
        # 1. Initial design: Evaluate a set of points via Latin Hypercube Sampling
        # to seed the surrogate models and establish initial objective ranges.
        initial_X = self._sample_points(self.n_initial_design)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Check if enough evaluations were made to proceed with model fitting.
        # This can happen if the budget is very small, e.g., less than n_initial_design.
        min_points_for_model = max(2, self.dim + 1) # Minimum points required for stable model training.
        if self.y is None or self.y.shape[0] < min_points_for_model:
            logging.warning(f"Budget exhausted ({self.n_evals}) during initial design or not enough points evaluated ({self.y.shape[0] if self.y is not None else 0}). Returning current non-dominated points.")
            if self.y is None or self.y.shape[0] == 0:
                # If no points evaluated at all, return empty arrays with correct shapes.
                return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 1), np.array([]).reshape(0, self.dim)
            # If some points were evaluated but not enough for full BO, return their Pareto front.
            is_efficient_mask = is_pareto_efficient(self.y)
            return self.y[is_efficient_mask], self.X[is_efficient_mask]

        # 2. Main Bayesian Optimization loop: Iteratively refine the search.
        while self.n_evals < self.budget:
            # Fit surrogate models using the current archive of evaluated points.
            # Models are fitted on negated objectives for internal minimization.
            self._fit_model(self.X, self.y_internal) 

            # Determine the number of points to select in the current batch.
            current_batch_size = min(self.batch_size, self.budget - self.n_evals) 
            if current_batch_size == 0: # No budget left for more evaluations.
                break
            
            # Select the next batch of points using the acquisition function and diversity strategy.
            next_X = self._select_next_points(batch_size=current_batch_size)
            
            # If _select_next_points could not return any point (e.g., due to internal errors
            # or an extremely small candidate pool/selection pool), stop optimization.
            if next_X.shape[0] == 0:
                logging.warning("No new points selected by acquisition function. Stopping optimization.")
                break

            # Evaluate the newly selected points using the black-box function.
            new_y = self._evaluate_points(func, next_X)
            if new_y.size == 0: # No points evaluated due to budget exhaustion during _evaluate_points.
                break

            # Update the archive with the new evaluations.
            self._update_eval_points(next_X, new_y)

        # 3. Return the final non-dominated (Pareto) front from the entire archive.
        # This uses the original objective values (maximization).
        if self.y is None or self.y.shape[0] == 0:
            # This case should ideally be caught by the earlier check, but as a safeguard.
            return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 1), np.array([]).reshape(0, self.dim)
            
        is_efficient_mask = is_pareto_efficient(self.y)
        F_pareto = self.y[is_efficient_mask]
        X_pareto = self.X[is_efficient_mask]

        return F_pareto, X_pareto

