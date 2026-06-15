import logging
import sys

import numpy as np
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.swarm_based.WOA import OriginalWOA
from mealpy.utils.space import FloatVar


class BaseOptimizer:
    def __init__(self, problem, n_particles, max_iter):
        if "func" not in problem:
            print("Error: 'problem' dictionary is missing 'func' key.", file=sys.stderr)
            sys.exit(1)

        self.func = problem["func"]
        self.min_bound = problem["min_bound"]
        self.max_bound = problem["max_bound"]
        self.D = problem["D"]
        self.n_particles = n_particles
        self.max_iter = max_iter

    def _initialize(self):
        self.x = self.min_bound + (self.max_bound - self.min_bound) * np.random.rand(
            self.n_particles, self.D
        )
        self.pbest_pos = np.copy(self.x)
        self.pbest_val = np.array([self.func(p) for p in self.pbest_pos])

        best_particle_idx = np.argmin(self.pbest_val)
        self.gbest_val = self.pbest_val[best_particle_idx]
        self.gbest_pos = np.copy(self.pbest_pos[best_particle_idx])

        self.convergence_history = []

    def _update_bests(self, i, f_val):
        if f_val < self.pbest_val[i]:
            self.pbest_val[i] = f_val
            self.pbest_pos[i] = np.copy(self.x[i])

            if f_val < self.gbest_val:
                self.gbest_val = f_val
                self.gbest_pos = np.copy(self.x[i])

    def run(self):
        raise NotImplementedError


class PSO(BaseOptimizer):
    def __init__(self, problem, n_particles, max_iter, w=0.729, c1=1.494, c2=1.494):
        super().__init__(problem, n_particles, max_iter)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v = np.zeros((self.n_particles, self.D))

    def run(self):
        self._initialize()
        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(self.D), np.random.rand(self.D)
                cognitive_v = self.c1 * r1 * (self.pbest_pos[i] - self.x[i])
                social_v = self.c2 * r2 * (self.gbest_pos - self.x[i])
                self.v[i] = self.w * self.v[i] + cognitive_v + social_v
                self.x[i] = self.x[i] + self.v[i]
                self.x[i] = np.clip(self.x[i], self.min_bound, self.max_bound)
                f_val = self.func(self.x[i])
                self._update_bests(i, f_val)
            self.convergence_history.append(self.gbest_val)
        return self.gbest_val, self.convergence_history


class QPSO(BaseOptimizer):
    def __init__(self, problem, n_particles, max_iter, beta=0.5):
        super().__init__(problem, n_particles, max_iter)
        self.beta = beta

    def run(self):
        self._initialize()
        for t in range(self.max_iter):
            mbest = np.mean(self.pbest_pos, axis=0)
            current_beta = 1.0 - (1.0 - self.beta) * (t / self.max_iter)

            for i in range(self.n_particles):
                phi = np.random.rand(self.D)
                p = phi * self.pbest_pos[i] + (1 - phi) * self.gbest_pos
                u = np.random.rand(self.D)
                u[u == 0] = 1e-10

                abs_diff = np.abs(mbest - self.x[i])
                log_term = -np.log(u)

                rand_check = np.random.rand(self.D) > 0.5
                self.x[i] = np.where(
                    rand_check,
                    p + current_beta * abs_diff * log_term,
                    p - current_beta * abs_diff * log_term,
                )

                self.x[i] = np.clip(self.x[i], self.min_bound, self.max_bound)
                f_val = self.func(self.x[i])
                self._update_bests(i, f_val)
            self.convergence_history.append(self.gbest_val)
        return self.gbest_val, self.convergence_history


class ACO(BaseOptimizer):
    def __init__(self, problem, n_particles, max_iter, q=0.1, xi=0.85):
        super().__init__(problem, n_particles, max_iter)
        self.q = q
        self.xi = xi

    def run(self):
        self._initialize()
        archive = np.copy(self.x)
        archive_f = np.array([self.func(p) for p in archive])

        m = self.n_particles
        ranks = np.arange(1, m + 1)
        weights = (1.0 / (self.q * m * np.sqrt(2 * np.pi))) * np.exp(
            -((ranks - 1) ** 2) / (2 * (self.q * m) ** 2)
        )
        weights /= np.sum(weights)

        for _ in range(self.max_iter):
            idx = np.argsort(archive_f)
            archive = archive[idx]
            archive_f = archive_f[idx]

            sigma = self.xi * np.abs(archive[-1] - archive[0]) + 1e-12

            new_solutions = []
            new_f = []

            for _ in range(m):
                k = np.random.choice(m, p=weights)
                center = archive[k]
                candidate = center + np.random.randn(self.D) * sigma
                candidate = np.clip(candidate, self.min_bound, self.max_bound)
                f_val = self.func(candidate)
                new_solutions.append(candidate)
                new_f.append(f_val)

            combined = np.vstack([archive, np.array(new_solutions)])
            combined_f = np.concatenate([archive_f, np.array(new_f)])

            best_idx = np.argsort(combined_f)[:m]
            archive = combined[best_idx]
            archive_f = combined_f[best_idx]

            self.x = np.copy(archive)
            self.pbest_pos = np.copy(archive)
            self.pbest_val = np.copy(archive_f)

            best_particle_idx = np.argmin(self.pbest_val)
            self.gbest_val = self.pbest_val[best_particle_idx]
            self.gbest_pos = np.copy(self.pbest_pos[best_particle_idx])

            self.convergence_history.append(self.gbest_val)

        return self.gbest_val, self.convergence_history


class ABC(BaseOptimizer):
    def __init__(self, problem, n_particles, max_iter, limit=30):
        super().__init__(problem, n_particles, max_iter)
        self.limit = limit

    def run(self):
        self._initialize()
        food_sources = np.copy(self.x[: self.n_particles // 2])
        food_f = np.array([self.func(fs) for fs in food_sources])
        trials = np.zeros(len(food_sources), dtype=int)

        for _ in range(self.max_iter):
            num_food = len(food_sources)

            for i in range(num_food):
                k = np.random.randint(num_food)
                while k == i:
                    k = np.random.randint(num_food)
                dim = np.random.randint(self.D)
                phi = np.random.uniform(-1, 1)

                candidate = np.copy(food_sources[i])
                candidate[dim] = candidate[dim] + phi * (
                    candidate[dim] - food_sources[k][dim]
                )
                candidate = np.clip(candidate, self.min_bound, self.max_bound)
                f_val = self.func(candidate)

                if f_val < food_f[i]:
                    food_sources[i] = candidate
                    food_f[i] = f_val
                    trials[i] = 0
                else:
                    trials[i] += 1

            fitness = 1.0 / (1.0 + food_f - np.min(food_f) + 1e-12)
            probs = fitness / np.sum(fitness)

            for _ in range(num_food):
                i = np.random.choice(num_food, p=probs)
                k = np.random.randint(num_food)
                while k == i:
                    k = np.random.randint(num_food)
                dim = np.random.randint(self.D)
                phi = np.random.uniform(-1, 1)

                candidate = np.copy(food_sources[i])
                candidate[dim] = candidate[dim] + phi * (
                    candidate[dim] - food_sources[k][dim]
                )
                candidate = np.clip(candidate, self.min_bound, self.max_bound)
                f_val = self.func(candidate)

                if f_val < food_f[i]:
                    food_sources[i] = candidate
                    food_f[i] = f_val
                    trials[i] = 0
                else:
                    trials[i] += 1

            for i in range(num_food):
                if trials[i] > self.limit:
                    food_sources[i] = self.min_bound + (
                        self.max_bound - self.min_bound
                    ) * np.random.rand(self.D)
                    food_f[i] = self.func(food_sources[i])
                    trials[i] = 0

            best_idx = np.argmin(food_f)
            self.gbest_val = food_f[best_idx]
            self.gbest_pos = np.copy(food_sources[best_idx])
            self.convergence_history.append(self.gbest_val)

        return self.gbest_val, self.convergence_history


class AntBioQPSO(BaseOptimizer):
    def __init__(
        self, problem, n_particles, max_iter, phi1, phi2, phi3, beta, evaporation_rate, pheromone_deposit
    ):
        super().__init__(problem, n_particles, max_iter)
        self.phi1, self.phi2, self.phi3 = phi1, phi2, phi3
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit

    def _initialize(self):
        super()._initialize()
        self.pheromone = np.ones(self.n_particles)
        self.local_leader_pos = np.copy(self.pbest_pos)
        for i in range(self.n_particles):
            self.update_local_leader(i)

    def _update_bests(self, i, f_val):
        pbest_improved = False
        if f_val < self.pbest_val[i]:
            self.pbest_val[i] = f_val
            self.pbest_pos[i] = np.copy(self.x[i])
            pbest_improved = True

            if f_val < self.gbest_val:
                self.gbest_val = f_val
                self.gbest_pos = np.copy(self.x[i])
        return pbest_improved

    def update_local_leader(self, i):
        left_idx = (i - 1) % self.n_particles
        right_idx = (i + 1) % self.n_particles

        neighbor_indices = [left_idx, i, right_idx]
        neighbor_pheromones = self.pheromone[neighbor_indices]

        best_neighbor_idx_local = np.argmax(neighbor_pheromones)
        best_idx_global = neighbor_indices[best_neighbor_idx_local]

        self.local_leader_pos[i] = self.pbest_pos[best_idx_global]

    def run(self):
        self._initialize()

        for _ in range(self.max_iter):
            self.pheromone *= 1 - self.evaporation_rate

            for i in range(self.n_particles):
                self.update_local_leader(i)

            mbest = np.mean(self.pbest_pos, axis=0)
            current_beta = 1.0 - (1.0 - self.beta) * (_ / self.max_iter)

            for i in range(self.n_particles):
                p = self.pbest_pos[i]
                g = self.gbest_pos
                b = self.local_leader_pos[i]

                phi_sum = self.phi1 + self.phi2 + self.phi3 or 1.0
                q = (self.phi1 * p + self.phi2 * g + self.phi3 * b) / phi_sum

                u = np.random.rand(self.D)
                u[u == 0] = 1e-10
                abs_diff = np.abs(mbest - self.x[i])
                log_term = -np.log(u)
                rand_check = np.random.rand(self.D) > 0.5
                self.x[i] = np.where(
                    rand_check,
                    q + current_beta * abs_diff * log_term,
                    q - current_beta * abs_diff * log_term,
                )

                self.x[i] = np.clip(self.x[i], self.min_bound, self.max_bound)
                f_val = self.func(self.x[i])

                pbest_improved = self._update_bests(i, f_val)

                if pbest_improved:
                    self.pheromone[i] += self.pheromone_deposit

            self.convergence_history.append(self.gbest_val)

        return self.gbest_val, self.convergence_history


class BeeBioQPSO(BaseOptimizer):
    def __init__(
        self, problem, n_particles, max_iter, phi1, phi2, phi3, beta, stagnation_limit
    ):
        super().__init__(problem, n_particles, max_iter)
        self.phi1, self.phi2, self.phi3 = phi1, phi2, phi3
        self.beta = beta
        self.stagnation_limit = stagnation_limit

    def _initialize(self):
        super()._initialize()
        self.stagnation_counter = np.zeros(self.n_particles, dtype=int)
        self.local_leader_pos = np.copy(self.pbest_pos)
        for i in range(self.n_particles):
            self.update_local_leader(i)

    def _update_bests(self, i, f_val):
        if f_val < self.pbest_val[i]:
            self.pbest_val[i] = f_val
            self.pbest_pos[i] = np.copy(self.x[i])
            self.stagnation_counter[i] = 0

            if f_val < self.gbest_val:
                self.gbest_val = f_val
                self.gbest_pos = np.copy(self.x[i])
        else:
            self.stagnation_counter[i] += 1

    def update_local_leader(self, i):
        left_idx = (i - 1) % self.n_particles
        right_idx = (i + 1) % self.n_particles

        neighbor_indices = [left_idx, i, right_idx]
        neighbor_fitness = self.pbest_val[neighbor_indices]

        best_neighbor_idx_local = np.argmin(neighbor_fitness)
        best_idx_global = neighbor_indices[best_neighbor_idx_local]

        self.local_leader_pos[i] = self.pbest_pos[best_idx_global]

    def run(self):
        self._initialize()

        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                self.update_local_leader(i)

            mbest = np.mean(self.pbest_pos, axis=0)
            current_beta = 1.0 - (1.0 - self.beta) * (_ / self.max_iter)

            for i in range(self.n_particles):
                p = self.pbest_pos[i]
                g = self.gbest_pos
                b = self.local_leader_pos[i]

                phi_sum = self.phi1 + self.phi2 + self.phi3 or 1.0
                q = (self.phi1 * p + self.phi2 * g + self.phi3 * b) / phi_sum

                u = np.random.rand(self.D)
                u[u == 0] = 1e-10
                abs_diff = np.abs(mbest - self.x[i])
                log_term = -np.log(u)
                rand_check = np.random.rand(self.D) > 0.5
                self.x[i] = np.where(
                    rand_check,
                    q + current_beta * abs_diff * log_term,
                    q - current_beta * abs_diff * log_term,
                )

                self.x[i] = np.clip(self.x[i], self.min_bound, self.max_bound)
                f_val = self.func(self.x[i])

                self._update_bests(i, f_val)

            for i in range(self.n_particles):
                if self.stagnation_counter[i] > self.stagnation_limit:
                    self.x[i] = self.min_bound + (
                        self.max_bound - self.min_bound
                    ) * np.random.rand(self.D)
                    f_val = self.func(self.x[i])

                    self.pbest_pos[i] = np.copy(self.x[i])
                    self.pbest_val[i] = f_val
                    self.stagnation_counter[i] = 0

                    if f_val < self.gbest_val:
                        self.gbest_val = f_val
                        self.gbest_pos = np.copy(self.x[i])

            self.convergence_history.append(self.gbest_val)

        return self.gbest_val, self.convergence_history


class MealpyOptimizer(BaseOptimizer):
    """Adapter wrapping a mealpy swarm optimizer for the BioQPSO experiment harness."""

    _mealpy_class = None

    def run(self):
        logging.getLogger("mealpy").setLevel(logging.WARNING)

        def obj_func(solution):
            return float(self.func(np.asarray(solution)))

        mealpy_problem = {
            "obj_func": obj_func,
            "lb": self.min_bound,
            "ub": self.max_bound,
            "minmax": "min",
            "bounds": [
                FloatVar(lb=self.min_bound, ub=self.max_bound) for _ in range(self.D)
            ],
            "log_to": None,
        }

        optimizer = self._mealpy_class(epoch=self.max_iter, pop_size=self.n_particles)
        agent = optimizer.solve(mealpy_problem)

        self.gbest_val = float(agent.target.fitness)
        self.gbest_pos = np.asarray(agent.solution, dtype=float)
        self.convergence_history = list(optimizer.history.list_global_best_fit)

        return self.gbest_val, self.convergence_history


class GWO(MealpyOptimizer):
    _mealpy_class = OriginalGWO


class WOA(MealpyOptimizer):
    _mealpy_class = OriginalWOA


ALGORITHMS_TO_TEST = {
    "PSO": {"class": PSO, "params": {"w": 0.729, "c1": 1.494, "c2": 1.494}},
    "QPSO": {"class": QPSO, "params": {"beta": 0.5}},
    "ACO": {"class": ACO, "params": {"q": 0.1, "xi": 0.85}},
    "ABC": {"class": ABC, "params": {"limit": 30}},
    "AntBioQPSO": {
        "class": AntBioQPSO,
        "params": {
            "phi1": 0.4,
            "phi2": 0.3,
            "phi3": 0.3,
            "beta": 0.5,
            "evaporation_rate": 0.1,
            "pheromone_deposit": 1.0,
        },
    },
    "BeeBioQPSO": {
        "class": BeeBioQPSO,
        "params": {
            "phi1": 0.4,
            "phi2": 0.3,
            "phi3": 0.3,
            "beta": 0.5,
            "stagnation_limit": 15,
        },
    },
    "GWO": {"class": GWO, "params": {}},
    "WOA": {"class": WOA, "params": {}},
}

