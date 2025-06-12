# Implements NSGA-II
import numpy as np
from deap.tools._hypervolume import hv
import random
from pymoo.indicators.hv import HV


class NSGAII:
    def __init__(self, pop_size, num_gen=100, crossover_prob=0.9, mutation_prob=0.4):
        self.pop_size = pop_size
        self.num_gen = num_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.last_scalar_reward = None

    def scalarize_objectives(self, f, weights=(0.7, 0.15, 0.15)):
        """
        Converts a multi-objective vector f = [SE, IL, EC] to a single scalar reward.
        Lower IL and EC are better, so we negate them.
        """
        se, il, ec = f
        w_se, w_il, w_ec = weights
        return w_se * se - w_il * il - w_ec * ec  # Maximize SE, minimize IL/EC

    def initialize_population(self, num_vars):

        population = []

        # ====== 1. Structured Individuals =======
        num_structured = int(0.3 * self.pop_size)  # 30% structured

        for _ in range(num_structured):
            individual = np.zeros(num_vars)

            # Even allocation pattern: SU assigned to every 2nd channel
            for i in range(num_vars):
                if i % 2 == 0:
                    individual[i] = 1  # SU assigned
            population.append(individual)

        # ====== 2. Random Individuals ===========
        num_random = self.pop_size - num_structured
        for _ in range(num_random):
            individual = np.random.randint(0, 2, size=num_vars)  # binary allocation
            population.append(individual)

        return np.array(population)
        # return np.random.rand(self.pop_size, num_vars)

    def evaluate_population(self, population, bandwidth, SINR, interference, power, data_rate):
        
        # === Ensure matching shapes ===
        bandwidth = bandwidth.reshape(1, -1)
        SINR = SINR.reshape(1, -1)
        interference = interference.reshape(1, -1)
        power = power.reshape(1, -1)
        data_rate = data_rate.reshape(1, -1)

        # Objective 1: Maximize Spectrum Efficiency
        f1 = np.sum(population[:, :bandwidth.shape[1]] * bandwidth * np.log2(1 + np.clip(SINR, 1e-8, None)), axis=1)

        # Objective 2: Minimize Interference
        f2 = np.sum(np.abs(population * interference * power), axis=1)  # ensure positive
        f2 = np.clip(f2, 1e-8, None)  # to ensure non negative interference

        # Objective 3: Minimize Power Consumption
        f3 = np.sum(np.abs(population * (power / np.clip(data_rate, 1e-8, None))), axis=1)
        f3 = np.clip(f3, 1e-8, None)

        # Normalize each objective explicitly
        f1_norm = (f1 - np.min(f1)) / (np.max(f1) - np.min(f1) + 1e-8)
        f2_norm = (f2 - np.min(f2)) / (np.max(f2) - np.min(f2) + 1e-8)
        f3_norm = (f3 - np.min(f3)) / (np.max(f3) - np.min(f3) + 1e-8)

        w1, w2, w3 = 0.7, 0.15, 0.15
        scalar_reward = (w1 * f1_norm) - (w2 * f2_norm) - (w3 * f3_norm)

        self.last_scalar_reward = scalar_reward  # store to access externally

        return np.vstack((f1_norm, f2_norm, f3_norm)).T

    def non_dominated_sort(self, fitness):
        num_individuals = len(fitness)
        ranks = np.zeros(num_individuals)
        domination_count = np.zeros(num_individuals)
        dominates = [[] for _ in range(num_individuals)]

        for i in range(num_individuals):
            for j in range(num_individuals):
                if all(fitness[i] <= fitness[j]) and any(fitness[i] < fitness[j]):
                    dominates[i].append(j)
                elif all(fitness[j] <= fitness[i]) and any(fitness[j] < fitness[i]):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                ranks[i] = 1

        return ranks

    def crowding_distance(self, fitness):
        distances = np.zeros(len(fitness))
        for m in range(fitness.shape[1]):
            sorted_indices = np.argsort(fitness[:, m])
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf

            for i in range(1, len(fitness) - 1):
                distances[sorted_indices[i]] += (fitness[sorted_indices[i + 1], m] - fitness[sorted_indices[i - 1], m])

        return distances

    def crossover(self, parent1, parent2):
        u = np.random.rand()
        beta = (2 * u) ** (1 / (2 + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (2 + 1))
        child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
        return np.clip(child1, 0, 1), np.clip(child2, 0, 1)

    def mutate(self, individual):
        u = np.random.rand()
        delta = (2 * u) ** (1 / (20 + 1)) - 1 if u <= 0.5 else 1 - (2 * (1 - u)) ** (1 / (20 + 1))
        return np.clip(individual + delta, 0, 1)

    def optimize(self, num_vars, B, SINR, I, P, R):
        convergence_curve = []
        population = self.initialize_population(B.shape[1])
        hv_per_gen = []

        for gen in range(self.num_gen):
            fitness = self.evaluate_population(population, B, SINR, I, P, R)
            fitness = np.array(fitness, dtype=np.float64)

            # === Compute HV this generation ===
            try:
                if fitness.ndim != 2:
                    raise ValueError(f"Fitness must be a 2D array, got shape {fitness.shape}")
                ref_point = np.max(fitness, axis=0) + 0.05  # Reference point worse than all individuals
                hv_calc = HV(ref_point=ref_point)
                hv_value = hv_calc(fitness)
            except Exception as e:
                hv_value = 0  # fallback if HV fails
                print(f"Hypervolume error in Gen {gen}: {e}")
            hv_per_gen.append(hv_value)

            # === Scalarize each objective triple to a reward value
            scalar_rewards = [self.scalarize_objectives(f) for f in fitness]
            best_scalar = max(scalar_rewards)  # or min(...) depending on your design
            convergence_curve.append(best_scalar)
            self.last_scalar_reward = scalar_rewards  # (you may already have this)

            ranks = self.non_dominated_sort(fitness)
            distances = self.crowding_distance(fitness)

            # Step 1: Sort by rank
            sorted_idx = np.argsort(ranks)

            # Step 2: Fill next generation using rank first, then crowding distance
            final_indices = []
            unique_ranks = np.unique(ranks)

            for r in unique_ranks:
                rank_mask = (ranks == r)
                candidates = np.where(rank_mask)[0]

                if len(final_indices) + len(candidates) <= self.pop_size:
                    final_indices.extend(candidates.tolist())
                else:
                     # Sort remaining candidates by crowding distance (descending)
                    cd_sorted = candidates[np.argsort(-distances[candidates])]
                    final_indices.extend(cd_sorted[:self.pop_size - len(final_indices)].tolist())
                    break

            selected_indices = final_indices
          
            
            #ranks = self.non_dominated_sort(fitness)
            #distances = self.crowding_distance(fitness)
            # Combine rank and crowding distance for better selection
            #selection_scores = ranks + 1 / (1 + distances)
            #selected_indices = np.argsort(selection_scores)[:self.pop_size]

            # Retain the best individuals from the previous generation
            elite_count = max(1, int(0.1 * self.pop_size))  # Keep top 10% best solutions
            elite_indices = np.argsort(ranks)[:elite_count]
            #selected_indices[:elite_count] = elite_indices  # Ensure elites are preserved
            elites = population[selected_indices]

            # Avoid duplicates by removing elites from selected_indices
            remaining_indices = [idx for idx in selected_indices if idx not in elite_indices]
            remaining_population = population[remaining_indices]

            # Fill remaining population slots
            non_elite_count = self.pop_size - elite_count
            selected_population = np.vstack((elites, remaining_population[:non_elite_count]))

            # Crossover & Mutation
            new_population = []
            for i in range(0, len(selected_population), 2):
                if i in range(0, len(selected_population)):
                    if np.random.rand() < self.crossover_prob:
                        child1, child2 = self.crossover(selected_population[i], selected_population[i + 1])
                        new_population.extend([child1, child2])
                    else:
                        new_population.extend([selected_population[i], selected_population[i + 1]])

            # === Adaptive Mutation: decay mutation probability over time ===
            current_mutation_prob = max(0.05, self.mutation_prob * (1 - gen / self.num_gen))
            population = np.array(
                [self.mutate(ind) if np.random.rand() < self.mutation_prob else ind for ind in new_population])

        # Final evaluation
        self.convergence_curve = convergence_curve
        final_population = population 
        final_fitness = self.evaluate_population(population, B, SINR, I, P, R)
        final_fitness = np.array(final_fitness, dtype=np.float64)

        return final_fitness, convergence_curve, scalar_rewards, hv_per_gen, final_population


def optimize_spectrum_allocation(env, B, SINR, I, P, R, pop_size, num_gen):
    num_vars = B.shape[1]  # Use bandwidth dimension to define variables

    # Ensure randomness
    np.random.seed(None)
    random.seed(None)

    nsga = NSGAII(pop_size=pop_size, num_gen=num_gen, crossover_prob=0.9, mutation_prob=0.4)
    pareto_population, convergence_data, scalar_rewards, hv_per_gen, final_population = nsga.optimize(num_vars, B, SINR, I, P, R)

    # Extract final Pareto-optimal solutions with additional parameters
   
    perturbed_fitness = [
        np.array(f) + np.random.uniform(0.0, 0.05, size=len(f)) for f in pareto_population
    ]

    # perturbed_fitness = fitness

    # Track convergence data (e.g., best fitness value per generation)
    convergence_data = nsga.convergence_curve

    # Compute average scalar reward from NSGA-II final population
    avg_weighted_reward = np.mean(nsga.last_scalar_reward)

    return perturbed_fitness, convergence_data, avg_weighted_reward, hv_per_gen, final_population


def extract_strict_pareto_front(fitness_array):
    fitness_array = np.array(fitness_array)
    is_dominated = np.zeros(len(fitness_array), dtype=bool)

    for i in range(len(fitness_array)):
        for j in range(len(fitness_array)):
            if i != j and np.all(fitness_array[j] <= fitness_array[i]) and np.any(fitness_array[j] < fitness_array[i]):
                is_dominated[i] = True
                break

    return fitness_array[~is_dominated]
