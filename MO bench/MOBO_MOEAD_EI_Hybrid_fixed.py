from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress ConvergenceWarning from sklearn GPs
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.gaussian_process"
)


# Helper function for non-dominated sorting (from MOBO_ParEGO_GP_Optimized)
def _get_non_dominated_front(Y: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask for non-dominated points in Y.
    Assumes minimization for all objectives.
    This implementation is O(N^2 * M) in worst-case, but is robust and uses vectorized NumPy operations.
    """
    if Y.shape[0] == 0:
        return np.array([], dtype=bool)

    # Ensure Y is 2D
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    n_points = Y.shape[0]
    is_nondominated = np.ones(n_points, dtype=bool)

    # Iterate through each point i
    for i in range(n_points):
        if is_nondominated[i]:  # Only check if not already marked as dominated
            # Check if Y[i] is dominated by any other point Y[j]
            # Y[j] dominates Y[i] if all objectives of Y[j] are less than or equal to Y[i]
            # AND at least one objective of Y[j] is strictly less than Y[i]

            # Vectorized comparison:
            # Check if any point 'j' dominates 'i'
            # (Y <= Y[i]) checks if all objectives of j are less than or equal to i (element-wise)
            # (Y < Y[i]) checks if at least one objective of j is strictly less than i (element-wise)

            # Boolean array indicating for each point 'j' if it dominates Y[i]
            dominates_i = np.all(Y <= Y[i], axis=1) & np.any(Y < Y[i], axis=1)

            # Exclude Y[i] itself from the dominance check
            dominates_i[i] = False

            if np.any(dominates_i):
                is_nondominated[i] = False
    return is_nondominated


class MOBO_MOEAD_EI_Hybrid_Fixed:
    """
    Multi-Objective Optimization algorithm combining MOEA/D evolutionary principles
    with Gaussian Process surrogate models and Augmented Tchebycheff Expected Improvement (EI)
    acquisition for efficient black-box optimization.

    Main ideas:
    - Hybrid approach: Combines population-based MOEA/D search with GP surrogates.
    - GP surrogates: A separate GP for each objective, trained on a sliding window of recent data.
    - Evolutionary operators: SBX crossover and Polynomial mutation to generate offspring.
    - Surrogate-assisted acquisition: Augmented Tchebycheff Expected Improvement (EI) is used.
      Offspring objectives are predicted by GPs, then EI is calculated for each candidate
      against each MOEA/D weight vector.
    - Batch Selection: Selects a batch of promising candidates based on their EI scores across subproblems.
    - MOEA/D population update: Evaluated points update the population based on Augmented Tchebycheff scalarization.
    - Dynamic normalization: Objectives are scaled to [0, 1] using observed ideal and nadir points.
    - Initial design: Latin Hypercube Sampling (LHS).
    - Pareto front tracking: Maintains a global archive of all evaluated non-dominated solutions.
    """

    def __init__(
        self,
        budget: int,
        dim: int,
        bounds: np.ndarray | None = None,
        n_init_ratio: float = 0.25,  # Fraction of budget for initial design
        population_size: int = 50,  # Number of subproblems/solutions in the population
        n_offspring_surrogate: int = 200,  # Number of offspring generated for surrogate evaluation
        batch_size: int = 5,  # Number of offspring to evaluate with true func per iteration
        n_model_points: int = 150,  # Sliding window size for GP training
        kernel_nu: float = 2.5,  # Nu parameter for Matern kernel (0.5, 1.5, 2.5)
        rho: float = 0.01,  # Small constant for augmented Tchebycheff
        gp_n_restarts_optimizer: int = 1,
    ):  # Number of restarts for GP hyperparameter optimization
        # Fixed problem parameters
        self.budget = budget
        self.dim = dim
        # bounds has shape (2, dim), bounds[0]: lower bound, bounds[1]: upper bound
        if bounds is None:
            self.bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        else:
            self.bounds = np.asarray(bounds, dtype=float)

        # Hyperparameters (tuned by SMAC, defined in Space)
        self.n_init_ratio = n_init_ratio
        self.population_size = population_size
        self.n_offspring_surrogate = n_offspring_surrogate
        self.batch_size = batch_size
        self.n_model_points = n_model_points
        self.kernel_nu = kernel_nu
        self.rho = rho
        self.gp_n_restarts_optimizer = gp_n_restarts_optimizer

        # The number of objectives (self.n_obj) is unknown a priori.
        self.n_obj: int | None = None

        # Population (X_pop, F_pop) and global archive (X_all, y_all)
        self.X_pop: np.ndarray | None = (
            None  # Current population inputs (true evaluated)
        )
        self.F_pop: np.ndarray | None = (
            None  # Current population objectives (true evaluated)
        )
        self.X_all: np.ndarray | None = None  # Stores all evaluated points (inputs)
        self.y_all: np.ndarray | None = None  # Stores all evaluated points (objectives)

        # External Archive for Pareto Front
        self.X_pareto: np.ndarray | None = None
        self.F_pareto: np.ndarray | None = None

        self.n_evals = 0  # the number of function evaluations

        # Initial design size (budget-aware, from MOBO_ParEGO_GP_Optimized)
        self.n_init = max(
            min(int(self.budget * self.n_init_ratio), 2 * self.dim + 1), 1
        )
        self.n_init = min(
            self.n_init, self.budget
        )  # Ensure n_init doesn't exceed budget

        # Fix: ensure population_size doesn't exceed n_init (initial population)
        self.population_size = min(self.population_size, self.n_init)

        # MOEA/D specific components
        self.weights: np.ndarray | None = None  # Weight vectors
        self.ideal_point: np.ndarray | None = (
            None  # Dynamically updated ideal point (min observed)
        )
        self.nadir_point: np.ndarray | None = (
            None  # Dynamically updated nadir point (max observed)
        )

        # Surrogate models and scaler
        self.models: list[GaussianProcessRegressor] = []
        self.scaler_X = StandardScaler()  # Scaler for input features X

        # Fixed genetic operator parameters (not tuned by SMAC to keep config space compact)
        self.crossover_prob = 0.9
        self.mutation_prob = 1.0 / dim if dim > 0 else 1.0
        self.eta_c = 20.0  # Distribution index for SBX crossover (fixed)
        self.eta_m = 20.0  # Distribution index for polynomial mutation (fixed)

    def _generate_weight_vectors(self, n_obj: int, population_size: int) -> np.ndarray:
        """
        Generates uniformly distributed weight vectors.
        For n_obj=2, uses a simple grid. For n_obj > 2, uses random Dirichlet samples.
        """
        if n_obj == 2:
            if population_size < 2:
                return np.array([[0.5, 0.5]])  # Handle edge case
            weights = np.array(
                [
                    [
                        i / (population_size - 1),
                        (population_size - 1 - i) / (population_size - 1),
                    ]
                    for i in range(population_size)
                ]
            )
        else:
            # For M > 2, generating a perfectly uniform grid of arbitrary size is complex.
            # Using random Dirichlet samples for simplicity and to ensure weights sum to 1.
            weights = np.random.dirichlet(np.ones(n_obj), size=population_size)

        # Ensure weights sum to 1 (Dirichlet already does this, but for grid, it's good practice)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        return weights

    def _augmented_tchebycheff_scalarization(
        self,
        objectives: np.ndarray,
        weights: np.ndarray,
        ideal_point: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """
        Calculates the Augmented Tchebycheff scalarized value for a given set of objectives.
        Assumes minimization. Objectives should be normalized relative to ideal_point (z*).
        The ideal_point here is expected to be the 'zero' point in the normalized space.
        """
        # For minimization, Augmented Tchebycheff is max(w_j * (f_j - z_j*)) + rho * sum(w_j * (f_j - z_j*))
        # The objectives passed here are already normalized relative to the true ideal point,
        # and ideal_point for this function call is np.zeros(self.n_obj).
        # So (objectives - ideal_point) is effectively just 'objectives' (the normalized values).

        weighted_objectives = weights * (objectives - ideal_point)

        if objectives.ndim == 1:
            max_term = np.max(weighted_objectives)
            sum_term = np.sum(weighted_objectives)
            return max_term + rho * sum_term
        else:  # objectives is (N, n_obj)
            max_term = np.max(weighted_objectives, axis=1)
            sum_term = np.sum(weighted_objectives, axis=1)
            return max_term + rho * sum_term

    def _polynomial_mutation(
        self, x: np.ndarray, lb: np.ndarray, ub: np.ndarray, eta_m: float
    ) -> np.ndarray:
        """
        Polynomial mutation operator.
        Applies mutation to each dimension with probability `self.mutation_prob`.
        """
        y = np.copy(x)

        mutate_mask = np.random.rand(self.dim) < self.mutation_prob

        if np.any(mutate_mask):
            delta1 = (y[mutate_mask] - lb[mutate_mask]) / (
                ub[mutate_mask] - lb[mutate_mask]
            )
            delta2 = (ub[mutate_mask] - y[mutate_mask]) / (
                ub[mutate_mask] - lb[mutate_mask]
            )

            mut_pow = 1.0 / (eta_m + 1.0)

            xy = np.random.rand(np.sum(mutate_mask))

            val = np.zeros_like(xy)
            val[xy <= 0.5] = (2.0 * xy[xy <= 0.5]) ** mut_pow - 1.0
            val[xy > 0.5] = 1.0 - (2.0 * (1.0 - xy[xy > 0.5])) ** mut_pow

            y[mutate_mask] += val * (ub[mutate_mask] - lb[mutate_mask])

            y = np.clip(y, lb, ub)  # Ensure bounds are respected
        return y

    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        eta_c: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulated Binary Crossover (SBX) operator.
        """
        offspring1 = np.copy(parent1)
        offspring2 = np.copy(parent2)

        # Apply crossover if random number is less than crossover_prob
        if np.random.rand() < self.crossover_prob:
            rand_val = np.random.rand(self.dim)

            beta = np.zeros_like(rand_val)
            mask_le_05 = rand_val <= 0.5
            mask_gt_05 = rand_val > 0.5

            beta[mask_le_05] = (2.0 * rand_val[mask_le_05]) ** (1.0 / (eta_c + 1.0))
            beta[mask_gt_05] = (1.0 / (2.0 * (1.0 - rand_val[mask_gt_05]))) ** (
                1.0 / (eta_c + 1.0)
            )

            # Apply SBX formulas
            offspring1_new = 0.5 * (
                ((parent1 + parent2) - beta * np.abs(parent2 - parent1))
            )
            offspring2_new = 0.5 * (
                ((parent1 + parent2) + beta * np.abs(parent2 - parent1))
            )

            offspring1 = np.clip(offspring1_new, lb, ub)
            offspring2 = np.clip(offspring2_new, lb, ub)

        return offspring1, offspring2

    def _sample_points(self, n_points: int, sampler_type: str = "lhs") -> np.ndarray:
        # Sample n_points candidate points efficiently within self.bounds.
        if n_points <= 0:
            return np.array([])

        lb = self.bounds[0]
        ub = self.bounds[1]

        if sampler_type == "lhs":
            sampler = qmc.LatinHypercube(
                d=self.dim, seed=np.random.randint(0, 2**32 - 1)
            )
            samples = sampler.random(n_points)
        elif sampler_type == "random":
            samples = np.random.rand(n_points, self.dim)
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")

        # Scale samples from [0,1] to bounds
        scaled_samples = qmc.scale(samples, lb, ub)
        return scaled_samples

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        # Fit a surrogate model on (X, y).
        # Use a sliding window of the most recent data
        if X.shape[0] > self.n_model_points:
            X_train = X[-self.n_model_points :]
            y_train = y[-self.n_model_points :]
        else:
            X_train = X
            y_train = y

        # Ensure enough data points for GP fitting
        if X_train.shape[0] < 2:  # GP needs at least 2 points to fit
            self.models = []  # Clear models if not enough data
            return

        # Scale X_train features. Re-fit scaler for the current window.
        self.scaler_X.fit(X_train)
        X_train_scaled = self.scaler_X.transform(X_train)

        self.models = []
        for i in range(self.n_obj):
            # Matern kernel with automatic length_scale and noise_level optimization
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=[1.0] * self.dim,
                length_scale_bounds=(1e-2, 1e2),
                nu=self.kernel_nu,
            ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2))

            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,  # Normalize target values for better fitting
                n_restarts_optimizer=self.gp_n_restarts_optimizer,  # Use hyperparameter
                random_state=42 + i,
            )  # Different random state for each GP for diversity
            try:
                gp.fit(X_train_scaled, y_train[:, i])
            except (
                ValueError
            ):  # Catch potential errors like singular matrices, retry with fewer restarts
                gp = GaussianProcessRegressor(
                    kernel=Matern(length_scale=[1.0] * self.dim, nu=self.kernel_nu)
                    + WhiteKernel(noise_level=1e-5),
                    normalize_y=True,
                    n_restarts_optimizer=0,  # Fallback to 0 restarts for robustness
                    random_state=42 + i,
                )
                gp.fit(X_train_scaled, y_train[:, i])
            self.models.append(gp)

    def _select_next_points(self, batch_size: int) -> np.ndarray:
        # Select the next points to evaluate using a surrogate-assisted MOEA/D approach with EI.

        if (
            self.X_pop is None
            or self.F_pop is None
            or self.X_all is None
            or self.y_all is None
        ):
            # This case should ideally not be hit after initial design.
            return self._sample_points(batch_size, sampler_type="lhs")

        # If no models are trained (e.g., not enough data points), use random sampling
        if not self.models or len(self.models) != self.n_obj:
            return self._sample_points(batch_size, sampler_type="random")

        X_offspring_generated = []
        # Generate `n_offspring_surrogate` offspring from current population
        for _ in range(self.n_offspring_surrogate // 2):  # Generate pairs
            # Select two distinct parents from the current population
            idx1, idx2 = np.random.choice(len(self.X_pop), 2, replace=False)
            parent1 = self.X_pop[idx1]
            parent2 = self.X_pop[idx2]

            # Apply crossover and mutation
            child1, child2 = self._sbx_crossover(
                parent1, parent2, self.bounds[0], self.bounds[1], self.eta_c
            )
            child1 = self._polynomial_mutation(
                child1, self.bounds[0], self.bounds[1], self.eta_m
            )
            child2 = self._polynomial_mutation(
                child2, self.bounds[0], self.bounds[1], self.eta_m
            )

            X_offspring_generated.extend([child1, child2])

        X_offspring_generated = np.array(
            X_offspring_generated[: self.n_offspring_surrogate]
        )

        if X_offspring_generated.shape[0] == 0:
            return self._sample_points(batch_size, sampler_type="random")

        # Predict objectives and uncertainties for offspring using GPs
        X_offspring_scaled = self.scaler_X.transform(X_offspring_generated)
        mu_offspring_list = []
        sigma_offspring_list = []
        for model in self.models:
            mu, sigma = model.predict(X_offspring_scaled, return_std=True)
            mu_offspring_list.append(mu)
            sigma_offspring_list.append(sigma)

        mu_offspring = np.array(
            mu_offspring_list
        ).T  # Shape (n_offspring_surrogate, n_obj)
        sigma_offspring = np.array(
            sigma_offspring_list
        ).T  # Shape (n_offspring_surrogate, n_obj)

        # Dynamic normalization for all currently evaluated points (for f_min_scalar)
        # Using self.y_all to determine ideal/nadir for normalization
        ideal_point_full_archive = np.min(self.y_all, axis=0)
        nadir_point_full_archive = np.max(self.y_all, axis=0)
        obj_range_full_archive = nadir_point_full_archive - ideal_point_full_archive
        obj_range_full_archive[
            obj_range_full_archive == 0
        ] = 1e-6  # Avoid division by zero

        # Normalize all past observed objectives (for f_min_scalar)
        y_all_normalized = (
            self.y_all - ideal_point_full_archive
        ) / obj_range_full_archive

        # Normalize predicted objectives for offspring (for mu_s and sigma_s)
        mu_offspring_normalized = (
            mu_offspring - ideal_point_full_archive
        ) / obj_range_full_archive
        sigma_offspring_normalized = (
            sigma_offspring / obj_range_full_archive
        )  # Sigma also scaled by range

        # Acquisition values matrix: (n_offspring_surrogate, population_size)
        ei_matrix = np.zeros((X_offspring_generated.shape[0], self.population_size))

        # Epsilon for EI calculation to avoid division by zero
        epsilon_ei = 1e-10

        # Iterate over each MOEA/D weight vector (subproblem)
        for j in range(self.population_size):
            w_j = self.weights[j]

            # Calculate f_min_scalar for this weight vector from observed data
            # Objectives are already normalized relative to ideal_point (0 in normalized space)
            f_min_scalar_j = np.min(
                self._augmented_tchebycheff_scalarization(
                    y_all_normalized, w_j, np.zeros(self.n_obj), self.rho
                )
            )

            # Calculate augmented Tchebycheff mean for all offspring candidates
            # Here, ideal_point for scalarization is np.zeros(self.n_obj) because mu_offspring_normalized is already relative to ideal_point_full_archive
            tcheby_mean_offspring = self._augmented_tchebycheff_scalarization(
                mu_offspring_normalized, w_j, np.zeros(self.n_obj), self.rho
            )

            # Approximate scalarized std: use std of the term that maximizes the Tchebycheff component
            weighted_mu_terms = w_j * (
                mu_offspring_normalized - np.zeros(self.n_obj)
            )  # (n_offspring_surrogate, n_obj)
            k_max_all = np.argmax(
                weighted_mu_terms, axis=1
            )  # Index of the max objective for each candidate
            sigma_scalar_offspring = sigma_offspring_normalized[
                np.arange(X_offspring_generated.shape[0]), k_max_all
            ]

            # Calculate Expected Improvement (EI) for this weight vector across all candidates
            u = np.zeros_like(sigma_scalar_offspring)
            ei_vals_for_w_j = np.zeros_like(sigma_scalar_offspring)

            mask_positive_sigma = sigma_scalar_offspring > epsilon_ei

            u[mask_positive_sigma] = (
                f_min_scalar_j - tcheby_mean_offspring[mask_positive_sigma]
            ) / sigma_scalar_offspring[mask_positive_sigma]
            ei_vals_for_w_j[mask_positive_sigma] = sigma_scalar_offspring[
                mask_positive_sigma
            ] * (
                u[mask_positive_sigma] * norm.cdf(u[mask_positive_sigma])
                + norm.pdf(u[mask_positive_sigma])
            )

            ei_matrix[:, j] = ei_vals_for_w_j

        # Select candidates for batch evaluation
        # For each weight vector, find the offspring that maximizes EI
        best_offspring_indices_per_subproblem = np.argmax(
            ei_matrix, axis=0
        )  # indices into X_offspring_generated

        # Get unique candidates from these best-per-subproblem selections
        unique_selected_indices = np.unique(best_offspring_indices_per_subproblem)

        # If there are more unique candidates than batch_size, we need to prioritize.
        # Prioritization: take the ones with the highest overall EI (max EI across any subproblem they were best for).
        if len(unique_selected_indices) > batch_size:
            max_ei_for_unique_candidates = np.max(
                ei_matrix[unique_selected_indices], axis=1
            )
            sorted_unique_indices = unique_selected_indices[
                np.argsort(max_ei_for_unique_candidates)[::-1]
            ]  # Sort descending
            selected_X_indices = sorted_unique_indices[:batch_size]
        else:
            selected_X_indices = unique_selected_indices
            # If not enough unique candidates, fill the rest with random candidates from original pool
            if len(selected_X_indices) < batch_size:
                remaining_slots = batch_size - len(selected_X_indices)
                all_indices = np.arange(X_offspring_generated.shape[0])
                available_indices = np.setdiff1d(all_indices, selected_X_indices)
                if len(available_indices) > 0:
                    additional_indices = np.random.choice(
                        available_indices,
                        min(remaining_slots, len(available_indices)),
                        replace=False,
                    )
                    selected_X_indices = np.concatenate(
                        (selected_X_indices, additional_indices)
                    )

        if len(selected_X_indices) == 0:  # Fallback if no candidates could be selected
            return self._sample_points(batch_size, sampler_type="random")

        return X_offspring_generated[selected_X_indices]

    def _evaluate_points(
        self, func: Callable[[np.ndarray], np.ndarray], X_to_consider: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Evaluate the points in X_to_consider.
        # This method must be the only place where func is called.
        # Respect the remaining budget: do not exceed self.budget evaluations in total.
        # Update self.n_evals by the actual number of function calls performed.
        # Return a tuple (evaluated_y, evaluated_X).

        if self.n_evals >= self.budget:
            return np.array([]), np.array([])  # No budget left

        num_points_to_eval = min(X_to_consider.shape[0], self.budget - self.n_evals)
        if num_points_to_eval == 0:
            return np.array([]), np.array([])

        X_evaluated = X_to_consider[
            :num_points_to_eval
        ]  # This is the actual X being evaluated

        # Clip points to self.bounds
        lb = self.bounds[0]
        ub = self.bounds[1]
        X_clipped = np.clip(X_evaluated, lb, ub)  # Use X_evaluated here

        y_results = []
        for x_point in X_clipped:  # Iterate over X_clipped
            y_val = func(x_point)
            if self.n_obj is None:
                self.n_obj = y_val.shape[0]
                # Initialize ideal/nadir points and GP models after n_obj is known
                self.ideal_point = np.full(self.n_obj, np.inf)
                self.nadir_point = np.full(self.n_obj, -np.inf)
                self.models = [None] * self.n_obj  # Initialize GP models list here

            if y_val.shape[0] != self.n_obj:
                raise ValueError(
                    f"Function returned {y_val.shape[0]} objectives, but expected {self.n_obj}."
                )
            y_results.append(y_val)

        self.n_evals += num_points_to_eval

        # Update ideal and nadir points based on new observations
        if len(y_results) > 0:
            new_y_array = np.array(y_results)
            self.ideal_point = np.minimum(self.ideal_point, np.min(new_y_array, axis=0))
            self.nadir_point = np.maximum(self.nadir_point, np.max(new_y_array, axis=0))

        return (
            np.array(y_results),
            X_evaluated,
        )  # Return the actual X that was evaluated

    def _update_eval_points(self, new_X: np.ndarray, new_F: np.ndarray):
        # Update the global archive and the MOEA/D population.

        if new_X.shape[0] == 0:
            return

        # Update global archive (all evaluated points)
        if self.X_all is None:
            self.X_all = new_X
            self.y_all = new_F
        else:
            self.X_all = np.vstack((self.X_all, new_X))
            self.y_all = np.vstack((self.y_all, new_F))

        # If this is the initial population, initialize X_pop, F_pop and MOEA/D specific components
        if self.X_pop is None:
            self.X_pop = new_X
            self.F_pop = new_F
            # Generate weights only once at the start
            self.weights = self._generate_weight_vectors(
                self.n_obj, self.population_size
            )
        else:  # Subsequent updates
            # Normalize objectives using current ideal and nadir points for comparison
            obj_range = self.nadir_point - self.ideal_point
            obj_range[obj_range == 0] = 1e-6  # Handle zero range

            # For each new offspring, try to update population members
            for x_offspring, f_offspring in zip(new_X, new_F):
                f_offspring_norm = (f_offspring - self.ideal_point) / obj_range

                # Iterate through all subproblems in the current population
                for subproblem_idx in range(self.population_size):
                    current_f_pop = self.F_pop[subproblem_idx]
                    current_f_pop_norm = (current_f_pop - self.ideal_point) / obj_range

                    w = self.weights[subproblem_idx]

                    # Calculate Augmented Tchebycheff values for offspring and current population member
                    # using the specific weight vector for this subproblem
                    # Here, ideal_point for scalarization is np.zeros(self.n_obj) because objectives are normalized
                    tcheby_offspring = self._augmented_tchebycheff_scalarization(
                        f_offspring_norm, w, np.zeros(self.n_obj), self.rho
                    )
                    tcheby_current_pop = self._augmented_tchebycheff_scalarization(
                        current_f_pop_norm, w, np.zeros(self.n_obj), self.rho
                    )

                    # If offspring is better for this subproblem, replace
                    if tcheby_offspring < tcheby_current_pop:
                        self.X_pop[subproblem_idx] = x_offspring
                        self.F_pop[subproblem_idx] = f_offspring

        # Always update external Pareto front archive (EP) with all evaluated points
        nondominated_mask = _get_non_dominated_front(self.y_all)
        self.F_pareto = self.y_all[nondominated_mask]
        self.X_pareto = self.X_all[nondominated_mask]

    def __call__(
        self, func: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        # Main optimization loop.

        # Initial design
        initial_X_sampled = self._sample_points(self.n_init, sampler_type="lhs")
        initial_y_evaluated, initial_X_evaluated = self._evaluate_points(
            func, initial_X_sampled
        )
        self._update_eval_points(initial_X_evaluated, initial_y_evaluated)

        # Main optimization loop
        while self.n_evals < self.budget:
            # Fit surrogate models on all available data
            self._fit_model(self.X_all, self.y_all)

            # Select next points using surrogate-assisted MOEA/D acquisition
            next_X_to_eval_raw = self._select_next_points(self.batch_size)

            # Evaluate the selected points with the true function
            next_y_evaluated, next_X_evaluated = self._evaluate_points(
                func, next_X_to_eval_raw
            )

            # If budget ran out during evaluation, next_y_evaluated might be empty
            if next_y_evaluated.shape[0] == 0:
                break

            # Update archive and population with new evaluations
            self._update_eval_points(next_X_evaluated, next_y_evaluated)

        # Return the final non-dominated front from the global archive
        return self.F_pareto, self.X_pareto
