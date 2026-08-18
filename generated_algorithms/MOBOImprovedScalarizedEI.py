import numpy as np
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from collections.abc import Callable
import warnings
# from scipy.optimize import minimize # Removed for runtime improvement

# Suppress sklearn warnings for GP optimizer convergence
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


class MOBOImprovedScalarizedEI:
    """
    Multi-Objective Bayesian Optimization (MOBO) algorithm using Gaussian Process
    surrogates and a refined Evolutionary Algorithm (EA) to optimize a
    scalarized Expected Improvement (EI) acquisition function.
    This algorithm focuses on computational efficiency by optimizing EI for a
    single, dynamically chosen weighted sum scalarization per BO iteration,
    and systematically exploring the Pareto front by cycling through uniformly
    distributed weight vectors. It prioritizes robustness, scale-invariance,
    and Hypervolume maximization.
    """

    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None,
                 n_initial_design_points: int = -1,  # Special value to calculate default
                 model_window_size: int = 100,  # Tunable
                 gp_restarts_optimizer: int = 0,  # Tunable (0 or 1)
                 ea_pop_size: int = 50,  # Tunable
                 ea_generations: int = 25,  # Tunable
                 ea_mutation_rate: float = 0.1,  # Tunable
                 ea_crossover_rate: float = 0.9,  # Tunable
                 n_uniform_weights: int = 50,  # Tunable: Number of uniform weight vectors to cycle through
                 random_seed: int = 42):
        """
        Initializes the MOBOImprovedScalarizedEI optimizer.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
            bounds (np.ndarray | None): An array of shape (2, dim) specifying the
                                       lower and upper bounds for each dimension.
                                       bounds[0] is lower, bounds[1] is upper.
            n_initial_design_points (int): Number of points for the initial design phase.
                                           If -1, a budget-aware default is used.
            model_window_size (int): The number of most recent evaluations to use
                                     for training the Gaussian Process models.
            gp_restarts_optimizer (int): Number of restarts for the GP kernel's
                                         hyperparameter optimizer (0 or 1 to manage runtime).
            ea_pop_size (int): Population size for the internal Evolutionary Algorithm.
            ea_generations (int): Number of generations for the internal Evolutionary Algorithm.
            ea_mutation_rate (float): Probability of mutation for an individual in the EA.
            ea_crossover_rate (float): Probability of crossover between two parents in the EA.
            n_uniform_weights (int): The number of uniformly distributed weight vectors
                                     to pre-generate and cycle through for scalarization.
            random_seed (int): Seed for the random number generator for reproducibility.
        """
        # Fixed problem parameters
        self.budget = budget
        self.dim = dim
        if bounds is None:
            self.bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        else:
            self.bounds = np.asarray(bounds, dtype=float)

        # Hyperparameters (tuned by SMAC, defined in Space)
        if n_initial_design_points == -1:
            # Budget-aware default: min(max(2*dim+1, 5), budget//4)
            self.n_init = max(min(2 * self.dim + 1, self.budget // 4), 5)
        else:
            self.n_init = n_initial_design_points
        self.n_init = min(self.n_init, self.budget - 1 if self.budget > 1 else 1)
        if self.n_init <= 0:
            self.n_init = 1

        self.model_window_size = model_window_size
        self.gp_restarts_optimizer = gp_restarts_optimizer
        self.ea_pop_size = ea_pop_size
        self.ea_generations = ea_generations
        self.ea_mutation_rate = ea_mutation_rate
        self.ea_crossover_rate = ea_crossover_rate
        self.n_uniform_weights = n_uniform_weights

        # The number of objectives (self.n_obj) is unknown a priori.
        # It MUST be inferred on the first call to func inside _evaluate_points.
        self.n_obj: int | None = None

        # X has shape (n_points, n_dims), y has shape (n_points, n_obj)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.n_evals = 0  # The number of function evaluations

        # Store GP models, one for each objective
        self.gp_models: list[GaussianProcessRegressor] = []

        # Store current non-dominated front for returning results
        self.F_pareto: np.ndarray | None = None
        self.X_pareto: np.ndarray | None = None

        # Store min/max observed objective values for dynamic normalization
        self.y_min_observed: np.ndarray | None = None
        self.y_max_observed: np.ndarray | None = None
        self.y_norm_all: np.ndarray | None = None # Store normalized y values for efficiency

        # Store pre-generated uniformly distributed weight vectors
        self._uniform_weights: np.ndarray | None = None

        # Internal state for acquisition function optimization (set per BO iteration)
        self._current_acquisition_weight: np.ndarray | None = None
        self._current_acquisition_f_best_scalar: float | None = None

        # Random state for reproducibility of sampling (e.g., weights, GP random_state, EA)
        self.rng = np.random.default_rng(seed=random_seed)

    def _sample_points(self, n_points: int) -> np.ndarray:
        """
        Samples n_points candidate points efficiently within self.bounds using Sobol sequences.

        Args:
            n_points (int): The number of points to sample.

        Returns:
            np.ndarray: An array of shape (n_points, n_dims) containing the sampled points.
        """
        if n_points <= 0:
            return np.array([]).reshape(0, self.dim)
        # Use rng.integers for seed to ensure different Sobol sequences if called multiple times
        # and for reproducibility from self.rng.
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng.integers(0, 2**30))
        # Generate points in [0, 1]^dim
        points_unit = sampler.random(n_points)
        # Scale to actual bounds
        lb = self.bounds[0]
        ub = self.bounds[1]
        points = qmc.scale(points_unit, lb, ub)
        return points

    def _generate_uniform_weights(self, n_obj: int, n_weights: int) -> np.ndarray:
        """
        Generates n_weights uniformly distributed on the (n_obj-1)-simplex using Sobol sequences.
        """
        if n_obj == 1:
            return np.ones((n_weights, 1))

        # Generate points in [0,1]^n_obj and normalize to sum to 1
        sampler = qmc.Sobol(d=n_obj, scramble=True, seed=self.rng.integers(0, 2**30))
        weights_unit_cube = sampler.random(n_weights)
        weights = weights_unit_cube / np.sum(weights_unit_cube, axis=1, keepdims=True)
        return weights

    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fits independent Gaussian Process surrogate models for each objective
        on the provided training data.
        Updates min/max observed objective values and stores normalized y_all.

        Args:
            X_train (np.ndarray): Training input features (shape: n_samples, n_dims).
            y_train (np.ndarray): Training objective values (shape: n_samples, n_obj).
        """
        self.gp_models = []
        if self.X is None or self.X.shape[0] < 2:  # Need at least 2 points for GP fitting
            return # Models cannot be fitted meaningfully, _acquisition_function_for_ea will handle this

        # Update min/max observed for normalization using ALL historical data
        self.y_min_observed = np.min(self.y, axis=0)
        self.y_max_observed = np.max(self.y, axis=0)

        # Normalize ALL historical y data for internal use (e.g., f_best_scalar)
        y_range_all = self.y_max_observed - self.y_min_observed + 1e-6 # Add epsilon for stability
        self.y_norm_all = (self.y - self.y_min_observed) / y_range_all

        # Normalize y_train (windowed data) for GP fitting to the [0, 1] range
        # Use the global min/max observed for normalization consistency
        y_train_norm = (y_train - self.y_min_observed) / y_range_all

        for i in range(self.n_obj):
            # Matern kernel with nu=2.5 provides a good balance of smoothness.
            # ConstantKernel for amplitude, WhiteKernel for observation noise.
            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * \
                     Matern(length_scale=np.ones(self.dim), nu=2.5, length_scale_bounds=(1e-2, 1e2)) + \
                     WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-3))
            gp = GaussianProcessRegressor(kernel=kernel,
                                          alpha=1e-6,  # Jitter for numerical stability
                                          n_restarts_optimizer=self.gp_restarts_optimizer,
                                          normalize_y=False, # We handle normalization manually
                                          random_state=self.rng.integers(0, 2**30)) # Use RNG for GP's random state
            try:
                gp.fit(X_train, y_train_norm[:, i])
            except Exception:
                # If GP fitting with optimization fails (e.g., singular matrix), try without optimization.
                gp_fallback = GaussianProcessRegressor(kernel=kernel,
                                                       alpha=1e-6,
                                                       n_restarts_optimizer=0, # No restarts for fallback
                                                       normalize_y=False,
                                                       random_state=self.rng.integers(0, 2**30))
                gp_fallback.fit(X_train, y_train_norm[:, i])
                gp = gp_fallback # Use the fallback model
            self.gp_models.append(gp)

    def _acquisition_function_for_ea(self, x: np.ndarray) -> float:
        """
        Calculates the Expected Improvement (EI) for a single candidate point x
        based on a single weighted sum of normalized objectives. This serves as the
        fitness function for the internal Evolutionary Algorithm. Higher scores are better.

        The specific weight vector and f_best_scalar are set externally before the EA runs.

        Args:
            x (np.ndarray): A single candidate point (shape: n_dims,).

        Returns:
            float: The EI acquisition score for the given point and current weight vector.
        """
        # If models are not fitted or not enough data, return random score for exploration
        if not self.gp_models or self.X is None or self.X.shape[0] < 2 or \
           self.n_obj is None or self._current_acquisition_weight is None or \
           self._current_acquisition_f_best_scalar is None:
            return self.rng.random() # Return random score to encourage exploration in early stages

        # Reshape x for prediction (1, self.dim)
        x_reshaped = x.reshape(1, -1)
        
        w = self._current_acquisition_weight
        f_best_scalar = self._current_acquisition_f_best_scalar

        # Predict means and stds for all objectives for this candidate point
        mu_objs_norm = np.zeros(self.n_obj)
        sigma_objs_norm = np.zeros(self.n_obj)
        for i, gp in enumerate(self.gp_models):
            mu_objs_norm[i], sigma_objs_norm[i] = gp.predict(x_reshaped, return_std=True)

        # Ensure standard deviations are non-negative and prevent numerical issues
        sigma_objs_norm = np.maximum(1e-10, sigma_objs_norm)

        # Scalarize the predicted mean and variance using the current weight vector
        mu_scalar = np.sum(w * mu_objs_norm)
        sigma_scalar = np.sqrt(np.sum(w**2 * sigma_objs_norm**2)) # Assuming independence of GPs

        # Calculate Expected Improvement (EI)
        # For minimization, f_best is the minimum observed scalarized value.
        
        # Avoid division by zero for sigma_scalar
        if sigma_scalar < 1e-10:
            # If uncertainty is negligible, EI is 0 if no improvement, otherwise 0
            ei_val = 0.0 if mu_scalar >= f_best_scalar else 0.0
        else:
            Z = (f_best_scalar - mu_scalar) / sigma_scalar
            ei_val = (f_best_scalar - mu_scalar) * norm.cdf(Z) + \
                     sigma_scalar * norm.pdf(Z)
        
        # EI must be non-negative
        return float(np.maximum(0.0, ei_val))

    def _tournament_selection(self, population: np.ndarray, fitness_scores: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Performs tournament selection to choose parents.

        Args:
            population (np.ndarray): The current population.
            fitness_scores (np.ndarray): Fitness scores for each individual.
            k (int): Tournament size.

        Returns:
            np.ndarray: Selected parent.
        """
        if len(population) < k: # Handle cases where population is smaller than tournament size
            k = len(population)
        selection_indices = self.rng.choice(len(population), k, replace=False)
        tournament_fitness = fitness_scores[selection_indices]
        winner_index_in_tournament = np.argmax(tournament_fitness)
        return population[selection_indices[winner_index_in_tournament]]

    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray, eta: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulated Binary Crossover (SBX) operator.

        Args:
            p1 (np.ndarray): First parent.
            p2 (np.ndarray): Second parent.
            eta (float): Distribution index. Higher eta means children are closer to parents.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two offspring.
        """
        c1, c2 = p1.copy(), p2.copy()
        # Apply crossover with ea_crossover_rate probability for the entire pair
        if self.rng.random() < self.ea_crossover_rate:
            for i in range(self.dim):
                u = self.rng.random()
                if u <= 0.5:
                    beta = (2 * u)**(1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u)))**(1.0 / (eta + 1.0))
                
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        return c1, c2

    def _select_next_points(self, batch_size: int) -> np.ndarray:
        """
        Selects the next points to evaluate by optimizing the acquisition function
        using a refined Genetic Algorithm.

        Args:
            batch_size (int): The number of points to select. (Typically 1 for sequential BO).

        Returns:
            np.ndarray: An array of shape (batch_size, n_dims) containing the
                        selected points.
        """
        # If models are not fitted or not enough data, return random points for exploration
        if not self.gp_models or self.X is None or self.X.shape[0] < 2 or \
           self.n_obj is None or self.y_norm_all is None:
            return self._sample_points(batch_size)

        # Generate uniform weights if not already generated (only once after n_obj is known)
        if self._uniform_weights is None:
            self._uniform_weights = self._generate_uniform_weights(self.n_obj, self.n_uniform_weights)

        # Select a specific weight vector for this BO iteration by cycling through the uniform weights
        current_weight_idx = self.n_evals % self.n_uniform_weights
        self._current_acquisition_weight = self._uniform_weights[current_weight_idx]

        # Calculate the best observed scalarized value for this specific weight vector.
        # This is min(sum(w_m * y_norm_m)) over all observed normalized points.
        self._current_acquisition_f_best_scalar = np.min(
            np.sum(self._current_acquisition_weight * self.y_norm_all, axis=1)
        )

        # Initialize population using Sobol sequences within bounds
        population = self._sample_points(self.ea_pop_size)
        population = np.clip(population, self.bounds[0], self.bounds[1]) # Ensure initial population is within bounds

        # Add current Pareto front points to the initial population for better exploration
        if self.X_pareto is not None and self.X_pareto.shape[0] > 0:
            num_pareto_to_add = min(self.X_pareto.shape[0], self.ea_pop_size // 5) # Add up to 20% of pop from Pareto
            if num_pareto_to_add > 0:
                # Randomly sample from Pareto front to avoid bias
                pareto_indices = self.rng.choice(self.X_pareto.shape[0], num_pareto_to_add, replace=False)
                # Replace initial random points with Pareto points
                population[:num_pareto_to_add] = self.X_pareto[pareto_indices]

        best_individual_so_far = None
        best_fitness_so_far = -np.inf

        # Main EA loop
        for generation in range(self.ea_generations):
            fitness_scores = np.array([self._acquisition_function_for_ea(ind) for ind in population])

            # Update overall best individual
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness_so_far:
                best_fitness_so_far = fitness_scores[current_best_idx]
                best_individual_so_far = population[current_best_idx]

            new_population = []
            # Keep the best individual (elitism)
            if best_individual_so_far is not None:
                new_population.append(best_individual_so_far)

            while len(new_population) < self.ea_pop_size:
                # Selection (Tournament Selection)
                p1 = self._tournament_selection(population, fitness_scores)
                p2 = self._tournament_selection(population, fitness_scores)

                child1, child2 = p1.copy(), p2.copy() # Default to copies if no crossover

                # Crossover (SBX)
                child1, child2 = self._sbx_crossover(p1, p2)

                # Mutation (Gaussian perturbation)
                mutation_strength_factor = 0.1 # This can be a fixed factor or hyperparameter
                mutation_range = (self.bounds[1] - self.bounds[0]) * mutation_strength_factor

                if self.rng.random() < self.ea_mutation_rate:
                    child1 += self.rng.normal(0, mutation_range, self.dim)
                if self.rng.random() < self.ea_mutation_rate:
                    child2 += self.rng.normal(0, mutation_range, self.dim)

                # Clip children to bounds
                child1 = np.clip(child1, self.bounds[0], self.bounds[1])
                child2 = np.clip(child2, self.bounds[0], self.bounds[1])

                new_population.extend([child1, child2])
            
            population = np.array(new_population[:self.ea_pop_size]) # New population for next generation

        # Reset internal acquisition state for next iteration
        self._current_acquisition_weight = None
        self._current_acquisition_f_best_scalar = None

        # Return the best point found by EA
        if best_individual_so_far is None:
            # Fallback if no individual was found (e.g., ea_pop_size=0, very early exit)
            return self._sample_points(batch_size)
            
        return np.array([best_individual_so_far]) # Return as batch of size 1

    def _evaluate_points(self, func: Callable[[np.ndarray], np.ndarray], X_to_eval: np.ndarray) -> np.ndarray:
        """
        Evaluates the given points using the black-box function `func`.
        Handles budget tracking, objective inference, and archive updates.

        Args:
            func (Callable[[np.ndarray], np.ndarray]): The black-box objective function.
            X_to_eval (np.ndarray): An array of points to evaluate (shape: n_points, n_dims).

        Returns:
            np.ndarray: An array of objective values for the evaluated points (shape: n_points, n_obj).
        """
        if self.n_evals >= self.budget:
            return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0)

        num_points_to_eval = min(X_to_eval.shape[0], self.budget - self.n_evals)
        if num_points_to_eval == 0:
            return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0)

        actual_X_to_eval = X_to_eval[:num_points_to_eval]

        # Clip points to self.bounds before calling func, to respect the problem domain.
        lb = self.bounds[0]
        ub = self.bounds[1]
        X_clipped = np.clip(actual_X_to_eval, lb, ub)

        new_y_list = []
        for x_point in X_clipped:
            y_val = func(x_point)
            # Infer n_obj if not set from the first evaluation
            if self.n_obj is None:
                self.n_obj = len(y_val)
            # Ensure all evaluations return the consistent number of objectives
            if len(y_val) != self.n_obj:
                raise ValueError(f"Function returned {len(y_val)} objectives, but expected {self.n_obj}.")
            new_y_list.append(y_val)

        new_y = np.array(new_y_list)
        self.n_evals += num_points_to_eval

        # Update the archive with the new evaluations
        self._update_eval_points(actual_X_to_eval, new_y)

        return new_y

    def _get_pareto_front(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Identifies the non-dominated (Pareto) front from a set of points.
        Assumes minimization for all objectives.

        Args:
            X (np.ndarray): Input features of evaluated points.
            y (np.ndarray): Objective values of evaluated points.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple (X_pareto, F_pareto) containing
                                          the non-dominated solutions and their
                                          corresponding objective values.
        """
        if X.shape[0] == 0:
            return np.array([]).reshape(0, self.dim), np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0)

        is_nondominated = np.ones(y.shape[0], dtype=bool)
        for i in range(y.shape[0]):
            if is_nondominated[i]:  # Only check if not already marked as dominated
                # Create boolean arrays for comparison
                le = y <= y[i]  # True if y_j <= y_i for each objective
                lt = y < y[i]   # True if y_j < y_i for each objective

                # Check for domination: (all objectives of j are <= i) AND (at least one objective of j is < i)
                dominated_by_j_mask = np.all(le, axis=1) & np.any(lt, axis=1)
                
                # Exclude self-domination (a point cannot dominate itself)
                dominated_by_j_mask[i] = False
                
                if np.any(dominated_by_j_mask):
                    is_nondominated[i] = False

        return X[is_nondominated], y[is_nondominated]

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        """
        Updates the internal archive of evaluated points and the Pareto front.

        Args:
            new_X (np.ndarray): New input features to add to the archive.
            new_y (np.ndarray): New objective values to add to the archive.
        """
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update the Pareto front from the entire archive
        self.X_pareto, self.F_pareto = self._get_pareto_front(self.X, self.y)

    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Executes the main multi-objective optimization loop.

        Args:
            func (Callable[[np.ndarray], np.ndarray]): The black-box objective function.
                                                      Takes an array of shape (n_dims,)
                                                      and returns np.ndarray of shape (M,).

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple (F_pareto, X_pareto), where F_pareto
                                          has shape (K, n_obj) and X_pareto has shape
                                          (K, n_dims) for the final non-dominated set.
        """
        # 1. Initial Design: Evaluate a set of points to start building surrogate models.
        initial_X = self._sample_points(self.n_init)
        self._evaluate_points(func, initial_X)

        # Main Bayesian Optimization loop
        while self.n_evals < self.budget:
            # Determine the sliding window of recent data for model training
            start_idx = max(0, self.X.shape[0] - self.model_window_size)
            X_train_window = self.X[start_idx:]
            y_train_window = self.y[start_idx:]

            # 2. Fit Surrogate Models: Train independent GPs on the windowed data.
            # Only fit if enough data points are available in the window
            if X_train_window.shape[0] >= 2:
                self._fit_model(X_train_window, y_train_window)
            else:
                # If not enough data (e.g., at the very start with small window_size),
                # skip model fitting; _acquisition_function_for_ea will return random scores.
                pass

            # 3. Select Next Points: Use the Evolutionary Algorithm to propose the next point(s).
            # We select one point at a time (batch_size=1) for simplicity and sequential optimization.
            next_X_to_eval = self._select_next_points(batch_size=1)

            # 4. Evaluate Points: Call the true objective function for the selected points.
            self._evaluate_points(func, next_X_to_eval)

        # Return the final non-dominated front
        if self.F_pareto is None or self.X_pareto is None or self.F_pareto.shape[0] == 0:
            # If no evaluations or no non-dominated points found, return empty arrays
            return np.array([]).reshape(0, self.n_obj if self.n_obj is not None else 0), \
                   np.array([]).reshape(0, self.dim)
        
        return self.F_pareto, self.X_pareto

