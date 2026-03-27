import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import json
from scipy import stats 
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    D = len(x)
    return 10 * D + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    D = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / D))
    term2 = -np.exp(sum2 / D)
    return term1 + term2 + 20 + np.e

def griewank(x):
    sum_term = np.sum(x**2 / 4000)
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1

def rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2)

def schwefel(x):
    x = np.asarray(x)
    D = len(x)
    return 418.9829 * D - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def levy(x):
    x = np.asarray(x)
    w = 1 + (x - 1) / 4.0
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    wi = w[:-1]
    term2 = np.sum((wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + np.pi)**2))
    return term1 + term2 + term3

def zakharov(x):
    x = np.asarray(x)
    i = np.arange(1, len(x) + 1)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * i * x)
    return sum1 + sum2**2 + sum2**4

def styblinski_tang(x):
    x = np.asarray(x)
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)

def dixon_price(x):
    x = np.asarray(x)
    i = np.arange(2, len(x) + 1)
    term1 = (x[0] - 1)**2
    term2 = np.sum(i * (2 * x[1:]**2 - x[:-1])**2)
    return term1 + term2


def make_feature_selection_problem():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    def fitness(x):
        x = np.asarray(x)
        mask = x >= 0.5
        selected = np.sum(mask)
        if selected == 0:
            return 1.0

        X_sel = X[:, mask]
        clf = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(clf, X_sel, y, cv=5)
        mean_acc = np.mean(scores)

        feature_frac = selected / 30.0
        return 0.7 * (1.0 - mean_acc) + 0.3 * feature_frac

    problem = {
        'func': fitness,
        'min_bound': 0.0,
        'max_bound': 1.0,
        'D': 30,
        'X': X,
        'y': y
    }
    return problem


PROBLEMS = {
    'sphere': {
        'func': sphere,
        'min_bound': -100,
        'max_bound': 100,
        'D': 30
    },
    'rastrigin': {
        'func': rastrigin,
        'min_bound': -5.12,
        'max_bound': 5.12,
        'D': 30
    },
    'ackley': {
        'func': ackley,
        'min_bound': -32.768,
        'max_bound': 32.768,
        'D': 30
    },
    'griewank': {
        'func': griewank,
        'min_bound': -600,
        'max_bound': 600,
        'D': 30
    },
    'rosenbrock': {
        'func': rosenbrock,
        'min_bound': -2.048,
        'max_bound': 2.048,
        'D': 30
    },
    'schwefel': {
        'func': schwefel,
        'min_bound': -500,
        'max_bound': 500,
        'D': 30
    },
    'levy': {
        'func': levy,
        'min_bound': -10,
        'max_bound': 10,
        'D': 30
    },
    'zakharov': {
        'func': zakharov,
        'min_bound': -5,
        'max_bound': 10,
        'D': 30
    },
    'styblinski_tang': {
        'func': styblinski_tang,
        'min_bound': -5,
        'max_bound': 5,
        'D': 30
    },
    'dixon_price': {
        'func': dixon_price,
        'min_bound': -10,
        'max_bound': 10,
        'D': 30
    }
}

class BaseOptimizer:
    def __init__(self, problem, n_particles, max_iter):
        if 'func' not in problem:
            print(f"Error: 'problem' dictionary is missing 'func' key.", file=sys.stderr)
            sys.exit(1)
            
        self.func = problem['func']
        self.min_bound = problem['min_bound']
        self.max_bound = problem['max_bound']
        self.D = problem['D']
        self.n_particles = n_particles
        self.max_iter = max_iter

    def _initialize(self):
        self.x = self.min_bound + (self.max_bound - self.min_bound) * np.random.rand(self.n_particles, self.D)
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
        raise NotImplementedError("The 'run' method must be implemented by the subclass.")

class PSO(BaseOptimizer):
    def __init__(self, problem, n_particles, max_iter, w=0.729, c1=1.494, c2=1.494):
        super().__init__(problem, n_particles, max_iter)
        self.w = w; self.c1 = c1; self.c2 = c2
        self.v = np.zeros((self.n_particles, self.D))

    def run(self):
        self._initialize()
        for t in range(self.max_iter):
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
                self.x[i] = np.where(rand_check,
                                     p + current_beta * abs_diff * log_term,
                                     p - current_beta * abs_diff * log_term)
                
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

        for t in range(self.max_iter):
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

            # Employed bees
            for i in range(num_food):
                k = np.random.randint(num_food)
                while k == i:
                    k = np.random.randint(num_food)
                dim = np.random.randint(self.D)
                phi = np.random.uniform(-1, 1)

                candidate = np.copy(food_sources[i])
                candidate[dim] = candidate[dim] + phi * (candidate[dim] - food_sources[k][dim])
                candidate = np.clip(candidate, self.min_bound, self.max_bound)
                f_val = self.func(candidate)

                if f_val < food_f[i]:
                    food_sources[i] = candidate
                    food_f[i] = f_val
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Onlooker bees
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
                candidate[dim] = candidate[dim] + phi * (candidate[dim] - food_sources[k][dim])
                candidate = np.clip(candidate, self.min_bound, self.max_bound)
                f_val = self.func(candidate)

                if f_val < food_f[i]:
                    food_sources[i] = candidate
                    food_f[i] = f_val
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Scout bees
            for i in range(num_food):
                if trials[i] > self.limit:
                    food_sources[i] = self.min_bound + (self.max_bound - self.min_bound) * np.random.rand(self.D)
                    food_f[i] = self.func(food_sources[i])
                    trials[i] = 0

            best_idx = np.argmin(food_f)
            self.gbest_val = food_f[best_idx]
            self.gbest_pos = np.copy(food_sources[best_idx])
            self.convergence_history.append(self.gbest_val)

        return self.gbest_val, self.convergence_history

class AntBioQPSO(BaseOptimizer):
    def __init__(self, problem, n_particles, max_iter, phi1, phi2, phi3, beta,
                 evaporation_rate, pheromone_deposit):
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
        
        for t in range(self.max_iter):
            self.pheromone *= (1 - self.evaporation_rate)

            for i in range(self.n_particles):
                self.update_local_leader(i)
                
            mbest = np.mean(self.pbest_pos, axis=0)
            current_beta = 1.0 - (1.0 - self.beta) * (t / self.max_iter)
            
            for i in range(self.n_particles):
                p = self.pbest_pos[i]
                g = self.gbest_pos
                b = self.local_leader_pos[i]
                
                phi_sum = self.phi1 + self.phi2 + self.phi3 or 1.0
                q = (self.phi1 * p + self.phi2 * g + self.phi3 * b) / phi_sum
                
                u = np.random.rand(self.D); u[u == 0] = 1e-10
                abs_diff = np.abs(mbest - self.x[i])
                log_term = -np.log(u)
                rand_check = np.random.rand(self.D) > 0.5
                self.x[i] = np.where(rand_check,
                                     q + current_beta * abs_diff * log_term,
                                     q - current_beta * abs_diff * log_term)

                self.x[i] = np.clip(self.x[i], self.min_bound, self.max_bound)
                f_val = self.func(self.x[i])
                
                pbest_improved = self._update_bests(i, f_val)
                
                if pbest_improved:
                    self.pheromone[i] += self.pheromone_deposit
            
            self.convergence_history.append(self.gbest_val)
            
        return self.gbest_val, self.convergence_history

class AdaptiveBioQPSO(AntBioQPSO):
    def __init__(
        self,
        problem,
        n_particles,
        max_iter,
        phi1,
        phi2,
        phi3,
        beta,
        evaporation_rate,
        pheromone_deposit,
        phi3_min=0.1,
        phi3_max=0.5,
        delta=0.02,
        stagnation_window=20,
    ):
        super().__init__(
            problem=problem,
            n_particles=n_particles,
            max_iter=max_iter,
            phi1=phi1,
            phi2=phi2,
            phi3=phi3,
            beta=beta,
            evaporation_rate=evaporation_rate,
            pheromone_deposit=pheromone_deposit,
        )
        self.phi3_min = phi3_min
        self.phi3_max = phi3_max
        self.delta = delta
        self.stagnation_window = stagnation_window

    def run(self):
        self._initialize()

        phi3_history = []
        stagnation_counter = 0
        prev_gbest_val = self.gbest_val

        for t in range(self.max_iter):
            self.pheromone *= (1 - self.evaporation_rate)

            for i in range(self.n_particles):
                self.update_local_leader(i)

            mbest = np.mean(self.pbest_pos, axis=0)
            current_beta = 1.0 - (1.0 - self.beta) * (t / self.max_iter)

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

            # Update global stagnation counter based on gbest improvement.
            if self.gbest_val < prev_gbest_val:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            prev_gbest_val = self.gbest_val

            # Feedback-driven adaptive phi3 scheduling.
            if stagnation_counter > self.stagnation_window:
                self.phi3 = min(self.phi3 + self.delta, self.phi3_max)
            else:
                self.phi3 = max(self.phi3 - self.delta, self.phi3_min)

            # Keep phi1:phi2 ratio fixed at 4:3 and enforce phi1+phi2+phi3=1.
            remaining = 1.0 - self.phi3
            self.phi1 = 4.0 * remaining / 7.0
            self.phi2 = 3.0 * remaining / 7.0

            phi3_history.append(self.phi3)

        return self.gbest_val, self.convergence_history, phi3_history

class BeeBioQPSO(BaseOptimizer):
    def __init__(self, problem, n_particles, max_iter, phi1, phi2, phi3, beta,
                 stagnation_limit):
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
        
        for t in range(self.max_iter):
            for i in range(self.n_particles):
                self.update_local_leader(i)
                
            mbest = np.mean(self.pbest_pos, axis=0)
            current_beta = 1.0 - (1.0 - self.beta) * (t / self.max_iter)

            for i in range(self.n_particles):
                p = self.pbest_pos[i]
                g = self.gbest_pos
                b = self.local_leader_pos[i]
                
                phi_sum = self.phi1 + self.phi2 + self.phi3 or 1.0
                q = (self.phi1 * p + self.phi2 * g + self.phi3 * b) / phi_sum
                
                u = np.random.rand(self.D); u[u == 0] = 1e-10
                abs_diff = np.abs(mbest - self.x[i])
                log_term = -np.log(u)
                rand_check = np.random.rand(self.D) > 0.5
                self.x[i] = np.where(rand_check,
                                     q + current_beta * abs_diff * log_term,
                                     q - current_beta * abs_diff * log_term)

                self.x[i] = np.clip(self.x[i], self.min_bound, self.max_bound)
                f_val = self.func(self.x[i])
                
                self._update_bests(i, f_val)
            
            for i in range(self.n_particles):
                if self.stagnation_counter[i] > self.stagnation_limit:
                    self.x[i] = self.min_bound + (self.max_bound - self.min_bound) * np.random.rand(self.D)
                    f_val = self.func(self.x[i])
                    
                    self.pbest_pos[i] = np.copy(self.x[i])
                    self.pbest_val[i] = f_val
                    self.stagnation_counter[i] = 0
                    
                    if f_val < self.gbest_val:
                        self.gbest_val = f_val
                        self.gbest_pos = np.copy(self.x[i])

            self.convergence_history.append(self.gbest_val)
            
        return self.gbest_val, self.convergence_history


class LinearAdaBioQPSO(AntBioQPSO):
    def __init__(
        self,
        problem,
        n_particles,
        max_iter,
        phi1,
        phi2,
        phi3,
        beta,
        evaporation_rate,
        pheromone_deposit,
        phi3_min=0.1,
        phi3_max=0.5,
    ):
        super().__init__(
            problem=problem,
            n_particles=n_particles,
            max_iter=max_iter,
            phi1=phi1,
            phi2=phi2,
            phi3=phi3,
            beta=beta,
            evaporation_rate=evaporation_rate,
            pheromone_deposit=pheromone_deposit,
        )
        self.phi3_min = phi3_min
        self.phi3_max = phi3_max

    def run(self):
        self._initialize()
        self.phi3_history = []

        for t in range(self.max_iter):
            phi3 = self.phi3_max - (self.phi3_max - self.phi3_min) * (
                t / self.max_iter
            )
            phi3 = float(np.clip(phi3, self.phi3_min, self.phi3_max))
            self.phi3 = phi3
            self.phi1 = (1.0 - self.phi3) * (4.0 / 7.0)
            self.phi2 = (1.0 - self.phi3) * (3.0 / 7.0)
            self.phi3_history.append(self.phi3)

            self.pheromone *= (1 - self.evaporation_rate)

            for i in range(self.n_particles):
                self.update_local_leader(i)

            mbest = np.mean(self.pbest_pos, axis=0)
            current_beta = 1.0 - (1.0 - self.beta) * (t / self.max_iter)

            for i in range(self.n_particles):
                p = self.pbest_pos[i]
                g = self.gbest_pos
                b = self.local_leader_pos[i]

                q = self.phi1 * p + self.phi2 * g + self.phi3 * b

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

        return self.gbest_val, self.convergence_history, self.phi3_history


class CosineAdaBioQPSO(AntBioQPSO):
    def __init__(
        self,
        problem,
        n_particles,
        max_iter,
        phi1,
        phi2,
        phi3,
        beta,
        evaporation_rate,
        pheromone_deposit,
        phi3_min=0.1,
        phi3_max=0.5,
    ):
        super().__init__(
            problem=problem,
            n_particles=n_particles,
            max_iter=max_iter,
            phi1=phi1,
            phi2=phi2,
            phi3=phi3,
            beta=beta,
            evaporation_rate=evaporation_rate,
            pheromone_deposit=pheromone_deposit,
        )
        self.phi3_min = phi3_min
        self.phi3_max = phi3_max

    def run(self):
        self._initialize()
        self.phi3_history = []

        for t in range(self.max_iter):
            phi3 = self.phi3_min + 0.5 * (self.phi3_max - self.phi3_min) * (
                1.0 + np.cos(np.pi * t / self.max_iter)
            )
            phi3 = float(np.clip(phi3, self.phi3_min, self.phi3_max))
            self.phi3 = phi3
            self.phi1 = (1.0 - self.phi3) * (4.0 / 7.0)
            self.phi2 = (1.0 - self.phi3) * (3.0 / 7.0)
            self.phi3_history.append(self.phi3)

            self.pheromone *= (1 - self.evaporation_rate)

            for i in range(self.n_particles):
                self.update_local_leader(i)

            mbest = np.mean(self.pbest_pos, axis=0)
            current_beta = 1.0 - (1.0 - self.beta) * (t / self.max_iter)

            for i in range(self.n_particles):
                p = self.pbest_pos[i]
                g = self.gbest_pos
                b = self.local_leader_pos[i]

                q = self.phi1 * p + self.phi2 * g + self.phi3 * b

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

        return self.gbest_val, self.convergence_history, self.phi3_history


class FeedbackAdaBioQPSO(AntBioQPSO):
    def __init__(
        self,
        problem,
        n_particles,
        max_iter,
        phi1,
        phi2,
        phi3,
        beta,
        evaporation_rate,
        pheromone_deposit,
        phi3_min=0.1,
        phi3_max=0.5,
        delta=0.02,
        stagnation_window=20,
    ):
        super().__init__(
            problem=problem,
            n_particles=n_particles,
            max_iter=max_iter,
            phi1=phi1,
            phi2=phi2,
            phi3=phi3,
            beta=beta,
            evaporation_rate=evaporation_rate,
            pheromone_deposit=pheromone_deposit,
        )
        self.phi3_min = phi3_min
        self.phi3_max = phi3_max
        self.delta = delta
        self.stagnation_window = stagnation_window

        self.phi3 = 0.5 * (self.phi3_max + self.phi3_min)
        self.phi1 = (1.0 - self.phi3) * (4.0 / 7.0)
        self.phi2 = (1.0 - self.phi3) * (3.0 / 7.0)

    def run(self):
        self._initialize()
        self.phi3_history = []

        gbest_stagnation_counter = 0
        prev_gbest_val = self.gbest_val

        for t in range(self.max_iter):
            if gbest_stagnation_counter > self.stagnation_window:
                self.phi3 = min(self.phi3 + self.delta, self.phi3_max)
            else:
                self.phi3 = max(self.phi3 - self.delta, self.phi3_min)

            self.phi1 = (1.0 - self.phi3) * (4.0 / 7.0)
            self.phi2 = (1.0 - self.phi3) * (3.0 / 7.0)
            self.phi3_history.append(self.phi3)

            self.pheromone *= (1 - self.evaporation_rate)

            for i in range(self.n_particles):
                self.update_local_leader(i)

            mbest = np.mean(self.pbest_pos, axis=0)
            current_beta = 1.0 - (1.0 - self.beta) * (t / self.max_iter)

            for i in range(self.n_particles):
                p = self.pbest_pos[i]
                g = self.gbest_pos
                b = self.local_leader_pos[i]

                q = self.phi1 * p + self.phi2 * g + self.phi3 * b

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

            if self.gbest_val < prev_gbest_val:
                gbest_stagnation_counter = 0
            else:
                gbest_stagnation_counter += 1
            prev_gbest_val = self.gbest_val

        return self.gbest_val, self.convergence_history, self.phi3_history


class PerParticleAdaBioQPSO(BeeBioQPSO):
    def __init__(
        self,
        problem,
        n_particles,
        max_iter,
        phi1,
        phi2,
        phi3,
        beta,
        stagnation_limit,
        phi3_min=0.1,
        phi3_max=0.5,
    ):
        super().__init__(
            problem=problem,
            n_particles=n_particles,
            max_iter=max_iter,
            phi1=phi1,
            phi2=phi2,
            phi3=phi3,
            beta=beta,
            stagnation_limit=stagnation_limit,
        )
        self.phi3_min = phi3_min
        self.phi3_max = phi3_max

    def run(self):
        self._initialize()
        self.phi3_history = []

        denom = float(self.stagnation_limit) if self.stagnation_limit else 1.0

        for t in range(self.max_iter):
            for i in range(self.n_particles):
                self.update_local_leader(i)

            # Per-particle phi3_i based on each particle's stagnation counter.
            frac = np.clip(self.stagnation_counter.astype(float) / denom, 0.0, 1.0)
            phi3_vec = self.phi3_min + (self.phi3_max - self.phi3_min) * frac
            self.phi3_history.append(float(np.mean(phi3_vec)))

            mbest = np.mean(self.pbest_pos, axis=0)
            current_beta = 1.0 - (1.0 - self.beta) * (t / self.max_iter)

            for i in range(self.n_particles):
                p = self.pbest_pos[i]
                g = self.gbest_pos
                b = self.local_leader_pos[i]

                phi3_i = float(phi3_vec[i])
                phi1_i = (1.0 - phi3_i) * (4.0 / 7.0)
                phi2_i = (1.0 - phi3_i) * (3.0 / 7.0)
                q = phi1_i * p + phi2_i * g + phi3_i * b

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
                    self.x[i] = self.min_bound + (self.max_bound - self.min_bound) * np.random.rand(
                        self.D
                    )
                    f_val = self.func(self.x[i])

                    self.pbest_pos[i] = np.copy(self.x[i])
                    self.pbest_val[i] = f_val
                    self.stagnation_counter[i] = 0

                    if f_val < self.gbest_val:
                        self.gbest_val = f_val
                        self.gbest_pos = np.copy(self.x[i])

            self.convergence_history.append(self.gbest_val)

        return self.gbest_val, self.convergence_history, self.phi3_history


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
    "LinearAda": {
        "class": LinearAdaBioQPSO,
        "params": {
            "phi1": 0.4,
            "phi2": 0.3,
            "phi3": 0.3,
            "beta": 0.5,
            "evaporation_rate": 0.1,
            "pheromone_deposit": 1.0,
            "phi3_min": 0.1,
            "phi3_max": 0.5,
        },
    },
    "CosineAda": {
        "class": CosineAdaBioQPSO,
        "params": {
            "phi1": 0.4,
            "phi2": 0.3,
            "phi3": 0.3,
            "beta": 0.5,
            "evaporation_rate": 0.1,
            "pheromone_deposit": 1.0,
            "phi3_min": 0.1,
            "phi3_max": 0.5,
        },
    },
    "FeedbackAda": {
        "class": FeedbackAdaBioQPSO,
        "params": {
            "phi1": 0.4,
            "phi2": 0.3,
            "phi3": 0.3,
            "beta": 0.5,
            "evaporation_rate": 0.1,
            "pheromone_deposit": 1.0,
            "phi3_min": 0.1,
            "phi3_max": 0.5,
            "delta": 0.02,
            "stagnation_window": 20,
        },
    },
    "PerParticleAda": {
        "class": PerParticleAdaBioQPSO,
        "params": {
            "phi1": 0.4,
            "phi2": 0.3,
            "phi3": 0.3,
            "beta": 0.5,
            "stagnation_limit": 15,
            "phi3_min": 0.1,
            "phi3_max": 0.5,
        },
    },
}

def run_experiment(problem, algorithms, n_runs=30, max_iter=1000, n_particles=50):
    print(f"\n--- Running Experiment ---")
    print(f"Problem: {problem['name']} (D={problem['D']})")
    print(f"Runs: {n_runs}, Iterations: {max_iter}, Particles: {n_particles}\n")
    
    results_stats = {}
    results_histories = {}
    results_raw_bests = {}
    results_phi3_histories = {}
    
    for name, config in algorithms.items():
        print(f"Running {name}...")
        algo_class = config['class']
        params = config['params']
        
        run_bests = []
        run_histories = []
        run_phi3_histories = []
        
        start_time = time.time()
        for r in range(n_runs):
            optimizer = algo_class(
                problem=problem, 
                n_particles=n_particles, 
                max_iter=max_iter, 
                **params
            )
            run_result = optimizer.run()
            if isinstance(run_result, tuple) and len(run_result) == 3:
                best_val, history, phi3_history = run_result
                run_phi3_histories.append(phi3_history)
            else:
                best_val, history = run_result
            run_bests.append(best_val)
            run_histories.append(history)
        
        end_time = time.time()
        
        results_stats[name] = {
            'Best': np.min(run_bests),
            'Worst': np.max(run_bests),
            'Mean': np.mean(run_bests),
            'StdDev': np.std(run_bests),
            'Time (s)': (end_time - start_time) / n_runs
        }
        
        results_histories[name] = np.mean(run_histories, axis=0)
        results_raw_bests[name] = run_bests
        results_phi3_histories[name] = run_phi3_histories if run_phi3_histories else None
    
    return results_stats, results_histories, results_raw_bests, results_phi3_histories

def print_results_table(stats_data): 
    print("\n--- Experimental Results ---")
    header = f"{'Algorithm':<16} | {'Mean':<12} | {'StdDev':<12} | {'Best':<12} | {'Worst':<12} | {'Avg Time (s)':<12}"
    print(header)
    print("-" * len(header))
    for name, data in stats_data.items():
        print(f"{name:<16} | {data['Mean']:<12.2e} | {data['StdDev']:<12.2e} | {data['Best']:<12.2e} | {data['Worst']:<12.2e} | {data['Time (s)']:<12.4f}")

def run_statistical_analysis(results_raw_bests, control_name='QPSO'):
    print("\n--- Statistical Significance (Wilcoxon & Friedman) ---")
    print(f"(Comparing against control: {control_name})")

    if control_name not in results_raw_bests:
        print(f"Control algorithm '{control_name}' not in results. Skipping.")
        return {}

    control_data = np.array(results_raw_bests[control_name])
    algo_names = list(results_raw_bests.keys())

    # Friedman test across all algorithms
    try:
        friedman_args = [np.array(results_raw_bests[name]) for name in algo_names]
        friedman_stat, friedman_p = stats.friedmanchisquare(*friedman_args)
    except ValueError:
        friedman_stat, friedman_p = np.nan, np.nan

    header = (
        f"{'Algorithm':<16} | {'Wilcoxon p-value':<18} | "
        f"{'Significant (p<0.05)?':<22} | {'Friedman chi2':<14} | {'Friedman p':<12}"
    )
    print(header)
    print("-" * len(header))

    p_values = {}

    for name, data in results_raw_bests.items():
        data = np.array(data)

        if name == control_name:
            wilcoxon_p = 1.0
            significant = "Control"
        else:
            if np.array_equal(data, control_data):
                wilcoxon_p = 1.0
            else:
                try:
                    _, wilcoxon_p = stats.wilcoxon(data, control_data, alternative='less')
                except ValueError:
                    wilcoxon_p = np.nan

            significant = "YES" if (not np.isnan(wilcoxon_p) and wilcoxon_p < 0.05) else "No"

        p_values[name] = wilcoxon_p

        print(
            f"{name:<16} | "
            f"{wilcoxon_p:<18.4e} | "
            f"{significant:<22} | "
            f"{friedman_stat:<14.4f} | "
            f"{friedman_p:<12.4e}"
        )

    return p_values

def plot_convergence(histories, problem_name):
    print(f"\n--- Plotting Convergence ({problem_name}) ---")
    plt.figure(figsize=(12, 8))
    for name, history in histories.items():
        plt.plot(history, label=name)
    
    plt.title(f'Mean Convergence Curve ({problem_name})')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.savefig(f"convergence_{problem_name}.png")
    print(f"Saved convergence plot to convergence_{problem_name}.png")
    plt.close()

def plot_phi3_evolution(phi3_histories, problem_name):
    # phi3_histories: {algorithm_name: list_of_phi3_history_lists} or None entries
    if not isinstance(phi3_histories, dict):
        return

    plt.figure(figsize=(12, 6))

    plotted_any = False
    for name, runs in phi3_histories.items():
        if runs is None:
            continue
        if not runs:
            continue

        mean_phi3 = np.mean(np.asarray(runs, dtype=float), axis=0)
        x = np.arange(len(mean_phi3))
        plt.plot(x, mean_phi3, linewidth=2.0, label=name)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    plt.axhline(
        0.3,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Fixed BioQPSO baseline",
    )
    plt.title(f"φ₃ weight evolution — {problem_name}")
    plt.xlabel("Iteration")
    plt.ylabel("φ₃ (bio-leader weight)")
    plt.ylim(0, 1)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.join("outputs", "adabioqpso")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"phi3_evolution_{problem_name}.png")
    plt.savefig(save_path)
    plt.close()

def run_sensitivity_analysis():
    print("\n--- Running Sensitivity Analysis (AntBioQPSO on sphere) ---")

    N_RUNS = 10
    MAX_ITER = 500
    N_PARTICLES = 30

    evaporation_rates = [0.05, 0.1, 0.2, 0.3, 0.5]
    pheromone_deposits = [0.5, 1.0, 2.0, 5.0, 10.0]
    phi3_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    problem = PROBLEMS['sphere'].copy()
    problem['name'] = 'sphere'

    def run_for_params(phi1, phi2, phi3, evaporation_rate, pheromone_deposit):
        run_bests = []
        for _ in range(N_RUNS):
            optimizer = AntBioQPSO(
                problem=problem,
                n_particles=N_PARTICLES,
                max_iter=MAX_ITER,
                phi1=phi1,
                phi2=phi2,
                phi3=phi3,
                beta=0.5,
                evaporation_rate=evaporation_rate,
                pheromone_deposit=pheromone_deposit
            )
            best_val, _ = optimizer.run()
            run_bests.append(best_val)
        run_bests = np.array(run_bests)
        return np.mean(run_bests), np.std(run_bests)

    # Baseline parameters (as used in ALGORITHMS_TO_TEST)
    base_phi1, base_phi2, base_phi3 = 0.4, 0.3, 0.3
    base_evaporation_rate = 0.1
    base_pheromone_deposit = 1.0

    evap_means, evap_stds = [], []
    for evap in evaporation_rates:
        mean_val, std_val = run_for_params(
            base_phi1, base_phi2, base_phi3, evap, base_pheromone_deposit
        )
        evap_means.append(mean_val)
        evap_stds.append(std_val)

    depo_means, depo_stds = [], []
    for depo in pheromone_deposits:
        mean_val, std_val = run_for_params(
            base_phi1, base_phi2, base_phi3, base_evaporation_rate, depo
        )
        depo_means.append(mean_val)
        depo_stds.append(std_val)

    phi3_means, phi3_stds = [], []
    for phi3 in phi3_values:
        remaining = 1.0 - phi3
        scale = remaining / (base_phi1 + base_phi2)
        phi1 = base_phi1 * scale
        phi2 = base_phi2 * scale
        mean_val, std_val = run_for_params(
            phi1, phi2, phi3, base_evaporation_rate, base_pheromone_deposit
        )
        phi3_means.append(mean_val)
        phi3_stds.append(std_val)

    print("\nSensitivity: evaporation_rate")
    header = f"{'evaporation_rate':<18} | {'Mean':<12} | {'StdDev':<12}"
    print(header)
    print("-" * len(header))
    for v, m, s in zip(evaporation_rates, evap_means, evap_stds):
        print(f"{v:<18.3f} | {m:<12.2e} | {s:<12.2e}")

    print("\nSensitivity: pheromone_deposit")
    header = f"{'pheromone_deposit':<18} | {'Mean':<12} | {'StdDev':<12}"
    print(header)
    print("-" * len(header))
    for v, m, s in zip(pheromone_deposits, depo_means, depo_stds):
        print(f"{v:<18.3f} | {m:<12.2e} | {s:<12.2e}")

    print("\nSensitivity: phi3 (bio-leader weight)")
    header = f"{'phi3':<18} | {'Mean':<12} | {'StdDev':<12}"
    print(header)
    print("-" * len(header))
    for v, m, s in zip(phi3_values, phi3_means, phi3_stds):
        print(f"{v:<18.3f} | {m:<12.2e} | {s:<12.2e}")

    # Plotting
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.errorbar(evaporation_rates, evap_means, yerr=evap_stds, fmt='-o', capsize=4)
    plt.title('Sensitivity of AntBioQPSO to evaporation_rate (sphere)')
    plt.xlabel('evaporation_rate')
    plt.ylabel('Best Fitness')
    plt.grid(True, ls="--", alpha=0.5)

    plt.subplot(3, 1, 2)
    plt.errorbar(pheromone_deposits, depo_means, yerr=depo_stds, fmt='-o', capsize=4)
    plt.title('Sensitivity of AntBioQPSO to pheromone_deposit (sphere)')
    plt.xlabel('pheromone_deposit')
    plt.ylabel('Best Fitness')
    plt.grid(True, ls="--", alpha=0.5)

    plt.subplot(3, 1, 3)
    plt.errorbar(phi3_values, phi3_means, yerr=phi3_stds, fmt='-o', capsize=4)
    plt.title('Sensitivity of AntBioQPSO to phi3 (sphere)')
    plt.xlabel('phi3')
    plt.ylabel('Best Fitness')
    plt.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("sensitivity_analysis.png")
    print("\nSaved sensitivity analysis plot to sensitivity_analysis.png")
    plt.close()

def _evaluate_feature_selection_solution(x, X, y):
    x = np.asarray(x)
    mask = x >= 0.5
    selected = int(np.sum(mask))
    if selected == 0:
        return 0.0, 0  # accuracy 0, 0 features

    X_sel = X[:, mask]
    clf = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(clf, X_sel, y, cv=5)
    mean_acc = float(np.mean(scores))
    return mean_acc, selected


if __name__ == "__main__":
    from bioqpso import (
        PROBLEMS,
        run_experiment,
        print_results_table,
        run_statistical_analysis,
        plot_convergence,
        run_sensitivity_analysis,
    )

    N_RUNS = 30
    MAX_ITER = 1000
    N_PARTICLES = 30

    output_dir = os.path.join("outputs", "adabioqpso")
    os.makedirs(output_dir, exist_ok=True)

    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        return obj

    run_metadata = {
        "output_dir": output_dir,
        "N_RUNS": N_RUNS,
        "MAX_ITER": MAX_ITER,
        "N_PARTICLES": N_PARTICLES,
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(_to_serializable(run_metadata), f, indent=2)

    for problem_name, problem_config in PROBLEMS.items():
        problem = problem_config.copy()
        problem["name"] = problem_name

        results_data, histories, raw_bests, phi3_histories = run_experiment(
            problem=problem,
            algorithms=ALGORITHMS_TO_TEST,
            n_runs=N_RUNS,
            max_iter=MAX_ITER,
            n_particles=N_PARTICLES,
        )

        print_results_table(results_data)
        p_values = run_statistical_analysis(raw_bests, control_name="QPSO")

        plot_convergence(histories, problem_name, output_dir=output_dir)
        plot_phi3_evolution(phi3_histories, problem_name)

        # Save numeric results and histories.
        with open(
            os.path.join(output_dir, f"{problem_name}_results_stats.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(_to_serializable(results_data), f, indent=2)

        with open(
            os.path.join(output_dir, f"{problem_name}_convergence_histories.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(_to_serializable(histories), f, indent=2)

        with open(
            os.path.join(output_dir, f"{problem_name}_raw_bests.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(_to_serializable(raw_bests), f, indent=2)

        with open(
            os.path.join(output_dir, f"{problem_name}_p_values.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(_to_serializable(p_values), f, indent=2)

        with open(
            os.path.join(output_dir, f"{problem_name}_phi3_histories.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(_to_serializable(phi3_histories), f, indent=2)

        print("\n" + "=" * 80 + "\n")

    run_sensitivity_analysis(output_dir=output_dir)

    print("\n--- Running Feature Selection Experiment (Breast Cancer) ---")
    fs_problem = make_feature_selection_problem()
    fs_problem["name"] = "feature_selection"

    N_RUNS_FS = 10
    MAX_ITER_FS = 200
    N_PARTICLES_FS = 30

    X_fs = fs_problem["X"]
    y_fs = fs_problem["y"]

    # Use the same algorithm configs as the main experiment, but evaluated on feature selection.
    FEATURE_SELECTION_ALGORITHMS = dict(ALGORITHMS_TO_TEST)
    # Explicitly ensure the 4 adaptive variants are included.
    for k in ["LinearAda", "CosineAda", "FeedbackAda", "PerParticleAda"]:
        if k in ALGORITHMS_TO_TEST:
            FEATURE_SELECTION_ALGORITHMS[k] = ALGORITHMS_TO_TEST[k]

    fs_stats = {}
    fs_histories = {}

    for name, config in FEATURE_SELECTION_ALGORITHMS.items():
        print(f"Running {name} on feature selection...")
        algo_class = config["class"]
        params = config["params"]

        run_accuracies = []
        run_features = []
        run_histories = []

        start_time = time.time()
        for _ in range(N_RUNS_FS):
            optimizer = algo_class(
                problem=fs_problem,
                n_particles=N_PARTICLES_FS,
                max_iter=MAX_ITER_FS,
                **params,
            )
            run_result = optimizer.run()
            if isinstance(run_result, tuple) and len(run_result) == 3:
                _, history, _phi3_history = run_result
            else:
                _, history = run_result
            run_histories.append(history)

            acc, n_feat = _evaluate_feature_selection_solution(
                optimizer.gbest_pos, X_fs, y_fs
            )
            run_accuracies.append(acc)
            run_features.append(n_feat)

        end_time = time.time()

        fs_stats[name] = {
            "MeanAcc": np.mean(run_accuracies),
            "StdAcc": np.std(run_accuracies),
            "AvgFeatures": np.mean(run_features),
            "Time (s)": (end_time - start_time) / N_RUNS_FS,
        }
        fs_histories[name] = np.mean(run_histories, axis=0)

    print("\n--- Feature Selection Results (Breast Cancer) ---")
    header = f"{'Algorithm':<16} | {'Mean Acc':<12} | {'Std Acc':<12} | {'Avg Features':<14} | {'Avg Time (s)':<12}"
    print(header)
    print("-" * len(header))
    for name, data in fs_stats.items():
        print(
            f"{name:<16} | "
            f"{data['MeanAcc']:<12.4f} | "
            f"{data['StdAcc']:<12.4f} | "
            f"{data['AvgFeatures']:<14.2f} | "
            f"{data['Time (s)']:<12.4f}"
        )

    plot_convergence(fs_histories, "feature_selection", output_dir=output_dir)

    results_path = os.path.join(output_dir, "feature_selection_results.json")
    feature_selection_results = {"stats": fs_stats, "mean_histories": fs_histories}
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(feature_selection_results), f, indent=2)
    print(f"Saved feature selection results to {results_path}")

    print("finished...")
