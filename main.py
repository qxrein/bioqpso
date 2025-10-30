import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy import stats 

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

def run_experiment(problem, algorithms, n_runs=30, max_iter=1000, n_particles=50):
    print(f"\n--- Running Experiment ---")
    print(f"Problem: {problem['name']} (D={problem['D']})")
    print(f"Runs: {n_runs}, Iterations: {max_iter}, Particles: {n_particles}\n")
    
    results_stats = {}
    results_histories = {}
    results_raw_bests = {}
    
    for name, config in algorithms.items():
        print(f"Running {name}...")
        algo_class = config['class']
        params = config['params']
        
        run_bests = []
        run_histories = []
        
        start_time = time.time()
        for r in range(n_runs):
            optimizer = algo_class(
                problem=problem, 
                n_particles=n_particles, 
                max_iter=max_iter, 
                **params
            )
            best_val, history = optimizer.run()
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
    
    return results_stats, results_histories, results_raw_bests

def print_results_table(stats_data): 
    print("\n--- Experimental Results ---")
    header = f"{'Algorithm':<16} | {'Mean':<12} | {'StdDev':<12} | {'Best':<12} | {'Worst':<12} | {'Avg Time (s)':<12}"
    print(header)
    print("-" * len(header))
    for name, data in stats_data.items():
        print(f"{name:<16} | {data['Mean']:<12.2e} | {data['StdDev']:<12.2e} | {data['Best']:<12.2e} | {data['Worst']:<12.2e} | {data['Time (s)']:<12.4f}")

def run_statistical_analysis(results_raw_bests, control_name='QPSO'):
    print("\n--- Statistical Significance (Mann-Whitney U) ---")
    print(f"(Comparing against control: {control_name})")
    
    if control_name not in results_raw_bests:
        print(f"Control algorithm '{control_name}' not in results. Skipping.")
        return
        
    control_data = results_raw_bests[control_name]
    
    header = f"{'Algorithm':<16} | {'p-value':<12} | {'Significant (p<0.05)?'}"
    print(header)
    print("-" * len(header))

    for name, data in results_raw_bests.items():
        if name == control_name:
            print(f"{name:<16} | {'(Control)':<12} | ...")
            continue
            
        try:
            stat, p_value = stats.mannwhitneyu(data, control_data, alternative='less')
            
            is_significant = "YES" if p_value < 0.05 else "No"
            print(f"{name:<16} | {p_value:<12.4e} | {is_significant}")
            
        except ValueError as e:
            print(f"{name:<16} | {'Test Error':<12} | {e}")

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


if __name__ == "__main__":
    
    N_RUNS = 30
    MAX_ITER = 1000
    N_PARTICLES = 30
    
    ALGORITHMS_TO_TEST = {
        'PSO': {
            'class': PSO,
            'params': {'w': 0.729, 'c1': 1.494, 'c2': 1.494}
        },
        'QPSO': {
            'class': QPSO,
            'params': {'beta': 0.5} 
        },
        'AntBioQPSO': {
            'class': AntBioQPSO,
            'params': {
                'phi1': 0.4, 'phi2': 0.3, 'phi3': 0.3, 'beta': 0.5,
                'evaporation_rate': 0.1,
                'pheromone_deposit': 1.0
            }
        },
        'BeeBioQPSO': {
            'class': BeeBioQPSO,
            'params': {
                'phi1': 0.4, 'phi2': 0.3, 'phi3': 0.3, 'beta': 0.5,
                'stagnation_limit': 15
            }
        }
    }

    for problem_name, problem_config in PROBLEMS.items():
        
        problem_config['name'] = problem_name
        
        
        results_data, histories, raw_bests = run_experiment(
            problem=problem_config,
            algorithms=ALGORITHMS_TO_TEST,
            n_runs=N_RUNS,
            max_iter=MAX_ITER,
            n_particles=N_PARTICLES
        )
        
        print_results_table(results_data)
        
        run_statistical_analysis(raw_bests, control_name='QPSO')
        
        plot_convergence(histories, problem_name)
        
        print("\n" + "="*80 + "\n")

    print("finished...")
