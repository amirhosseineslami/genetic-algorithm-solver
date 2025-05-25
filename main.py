import random
import numpy as np
from sympy import symbols

# --- Constants for 2-variable linear equations ---
POP_SIZE_INIT_POP_2VAR = 100
LOWER_BOUND_INIT_POP_2VAR = -100
UPPER_BOUND_INIT_POP_2VAR = 100
NUM_PARENTS_SELECTION_2VAR = 20
MUTATION_RATE_2VAR = 0.1
MUT_MIN_BOUND_2VAR = -20
MUT_MAX_BOUND_2VAR = 20
BEST_FITNESS_MAX_ERROR_2VAR = 1e-8

# --- Constants for 3-variable and 4-variable equations ---
INIT_POPULATION_SIZE_NVAR = 500
LOWER_BOUND_INIT_POP_NVAR = -100
UPPER_BOUND_INIT_POP_NVAR = 100
MUTATION_RATE_NVAR = 0.05
BEST_FITNESS_MAX_ERROR_NVAR = 1e-8
GENERATIONS_DEFAULT_AMOUNT_NVAR = 500
TOURNAMENT_DEFAULT_SIZE_NVAR = 25
CONVERGENCE_THRESHOLD_NVAR = 1e-6
CONVERGENCE_PATIENCE_NVAR = 50
X_AVOID_ZERO_THRESHOLD_NVAR = 1e-3  # to avoid x ≈ 0 in division

# --- 2-variable linear equation functions ---
def get_equation_coefficients():
    print("\nEnter coefficients for the first equation (a1 * x + b1 * y = c1):")
    a1 = float(input("input a1: "))
    b1 = float(input("input b1: "))
    c1 = float(input("input c1: "))

    print("\nEnter coefficients for the second equation (a2 * x + b2 * y = c2):")
    a2 = float(input("input a2: "))
    b2 = float(input("input b2: "))
    c2 = float(input("input c2: "))

    return (a1, b1, c1), (a2, b2, c2)

def generate_initial_population_2var(pop_size=POP_SIZE_INIT_POP_2VAR, lower_bound=LOWER_BOUND_INIT_POP_2VAR, upper_bound=UPPER_BOUND_INIT_POP_2VAR):
    population = []
    for _ in range(pop_size):
        x = random.uniform(lower_bound, upper_bound)
        y = random.uniform(lower_bound, upper_bound)
        chromosome = [x, y]
        population.append(chromosome)
    return population

def estimate_bounds_2var(eq1, eq2, scale=2):
    max_c = max(abs(eq1[2]), abs(eq2[2]))
    bound = max_c * scale
    return -bound, bound

def fitness_2var(chromosome, eq1, eq2):
    x, y = chromosome
    a1, b1, c1 = eq1
    a2, b2, c2 = eq2

    error1 = (a1 * x + b1 * y) - c1
    error2 = (a2 * x + b2 * y) - c2

    totalSqueredError = error1**2 + error2**2
    return totalSqueredError

def select_parents_2var(population, eq1, eq2, num_parents=NUM_PARENTS_SELECTION_2VAR):
    population_sorted = sorted(population, key=lambda chromo: fitness_2var(chromo, eq1, eq2))
    return population_sorted[:num_parents]

def crossover_2var(parent1, parent2):
    x = (parent1[0] + parent2[0]) / 2
    y = (parent1[1] + parent2[1]) / 2
    return [x, y]

def mutate_2var(chromosome, mutation_rate=MUTATION_RATE_2VAR):
    if random.random() < mutation_rate:
        chromosome[0] += random.uniform(MUT_MIN_BOUND_2VAR, MUT_MAX_BOUND_2VAR)

    if random.random() < mutation_rate:
        chromosome[1] += random.uniform(MUT_MIN_BOUND_2VAR, MUT_MAX_BOUND_2VAR)
    return chromosome

def genetic_algorithm_2var(eq1, eq2, generations=1000, pop_size=POP_SIZE_INIT_POP_2VAR):
    lower, upper = estimate_bounds_2var(eq1, eq2)
    population = generate_initial_population_2var(pop_size, lower, upper)

    for generation in range(generations):
        population = sorted(population, key=lambda chromo: fitness_2var(chromo, eq1, eq2))
        best_fitness = fitness_2var(population[0], eq1, eq2)

        if best_fitness < BEST_FITNESS_MAX_ERROR_2VAR:
            print(f"Solution found in generation {generation}")
            return population[0]

        parents = select_parents_2var(population, eq1, eq2, num_parents=NUM_PARENTS_SELECTION_2VAR)

        new_population = []
        while len(new_population) < pop_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = crossover_2var(parent1, parent2)
            child = mutate_2var(child)
            new_population.append(child)

        population = new_population

    print("No exact solution found. Best approximation:")
    return sorted(population, key=lambda chromo: fitness_2var(chromo, eq1, eq2))[0]

# --- N-variable equation functions (for 3 and 4 variables) ---
def parse_equation_nvar(equation_str, variables):
    if "=" not in equation_str:
        raise ValueError("Equation must contain an '=' sign.")

    left_expr, right_expr = equation_str.split("=")

    print(f"Parsing equation: {left_expr.strip()} = {right_expr.strip()}")

    try:
        # Create a dictionary of variables for eval
        var_dict = {str(v): v for v in variables}
        left_expr = eval(left_expr, var_dict)
        right_expr = eval(right_expr, var_dict)
    except Exception as e:
        raise ValueError(f"Invalid equation format: {e}")

    def eq_func(*vals):
        try:
            # Check for division by zero for the first variable (x)
            if abs(vals[0]) < X_AVOID_ZERO_THRESHOLD_NVAR:
                return float("inf")

            # Create a dictionary for subs based on the order of variables
            subs_dict = {variables[i]: vals[i] for i in range(len(variables))}
            result = left_expr.subs(subs_dict) - right_expr
            return float(result)
        except Exception:
            return float("inf")
    return eq_func

def generate_initial_population_nvar(num_vars, population_size=INIT_POPULATION_SIZE_NVAR, value_range=(LOWER_BOUND_INIT_POP_NVAR, UPPER_BOUND_INIT_POP_NVAR)):
    min_val, max_val = value_range
    population = np.random.uniform(min_val, max_val, size=(population_size, num_vars))
    for i in range(len(population)):
        # Ensure x (first variable) is not too close to zero
        while abs(population[i][0]) < X_AVOID_ZERO_THRESHOLD_NVAR:
            population[i][0] = np.random.uniform(min_val, max_val)
    return population

def calculate_fitness_nvar(population, equations):
    return np.array([
        np.sqrt(sum((eq(*chrom))**2 for eq in equations)) for chrom in population
    ])

def tournament_selection_nvar(population, fitness_values, tournament_size=TOURNAMENT_DEFAULT_SIZE_NVAR):
    selected = []
    for _ in range(len(population)):
        indices = np.random.choice(len(population), tournament_size, replace=False)
        winner = population[indices[np.argmin(fitness_values[indices])]]
        selected.append(winner)
    return np.array(selected)

def single_point_crossover_nvar(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        p1 = parents[i]
        p2 = parents[i+1] if i+1 < len(parents) else parents[i]
        point = np.random.randint(1, len(p1))
        offspring.append(np.concatenate((p1[:point], p2[point:])))
        offspring.append(np.concatenate((p2[:point], p1[point:])))
    return np.array(offspring)

def mutation_nvar(offspring, mutation_rate, value_range):
    min_val, max_val = value_range
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            if np.random.rand() < mutation_rate:
                delta = np.random.uniform(-1, 1)
                offspring[i][j] += delta
                offspring[i][j] = np.clip(offspring[i][j], min_val, max_val)
    return offspring

def genetic_algorithm_nvar(equations, num_vars, generations=GENERATIONS_DEFAULT_AMOUNT_NVAR,
                             population_size=INIT_POPULATION_SIZE_NVAR, value_range=(LOWER_BOUND_INIT_POP_NVAR, UPPER_BOUND_INIT_POP_NVAR),
                             tournament_size=TOURNAMENT_DEFAULT_SIZE_NVAR, mutation_rate=MUTATION_RATE_NVAR):

    population = generate_initial_population_nvar(num_vars, population_size, value_range)
    best_fitness_history = []

    for generation in range(generations):
        fitness_values = calculate_fitness_nvar(population, equations)
        best_fitness = np.min(fitness_values)
        best_individual = population[np.argmin(fitness_values)]
        best_fitness_history.append(best_fitness)

        if generation % 10 == 0:
            print(f"Generation {generation}: Fitness={best_fitness:.6f} | Solution={best_individual}")

        if best_fitness < BEST_FITNESS_MAX_ERROR_NVAR:
            print(f"\nSolution found in generation {generation}!")
            return best_individual, best_fitness

        if generation > CONVERGENCE_PATIENCE_NVAR:
            recent = best_fitness_history[-CONVERGENCE_PATIENCE_NVAR:]
            if np.std(recent) < CONVERGENCE_THRESHOLD_NVAR:
                print(f"\nConverged at generation {generation}")
                break

        selected = tournament_selection_nvar(population, fitness_values, tournament_size)
        offspring = single_point_crossover_nvar(selected)
        population = mutation_nvar(offspring, mutation_rate, value_range)

    best_index = np.argmin(fitness_values)
    best_solution = population[best_index]
    print(f"\nBest Solution Found: {best_solution} | Fitness: {best_fitness:.8f}")
    return best_solution, best_fitness

def get_equations_from_user_nvar(num_eqs, variables):
    equations_str = []
    print(f"\nEnter {num_eqs} equations with variables {', '.join(map(str, variables))}:")
    for i in range(num_eqs):
        equations_str.append(input(f"Equation {i+1}: "))
    return [parse_equation_nvar(eq_str, variables) for eq_str in equations_str]

# --- Main program flow ---
def main():
    while True:
        print("\n--- Equation Solver ---")
        print("1. Solve 2-variable linear equations (e.g., ax + by = c)")
        print("2. Solve 3-variable non-linear equations (e.g., x + y*z = c)")
        print("3. Solve 4-variable non-linear equations (e.g., x + y*z - t = c)")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            eq1, eq2 = get_equation_coefficients()
            solution = genetic_algorithm_2var(eq1, eq2)
            print(f"\nApproximate solution: x = {solution[0]:.6f}, y = {solution[1]:.6f}")
        elif choice == '2':
            x, y, z = symbols("x y z")
            variables = [x, y, z]
            num_eqs = 3
            equations = get_equations_from_user_nvar(num_eqs, variables)
            best_solution, best_fitness = genetic_algorithm_nvar(
                equations=equations,
                num_vars=num_eqs,
                population_size=INIT_POPULATION_SIZE_NVAR,
                generations=GENERATIONS_DEFAULT_AMOUNT_NVAR,
                value_range=(LOWER_BOUND_INIT_POP_NVAR, UPPER_BOUND_INIT_POP_NVAR),
                tournament_size=TOURNAMENT_DEFAULT_SIZE_NVAR,
                mutation_rate=MUTATION_RATE_NVAR
            )
            print(f"\nBest solution found: x={best_solution[0]:.6f}, y={best_solution[1]:.6f}, z={best_solution[2]:.6f}")
            print(f"Final Fitness: {best_fitness:.8f}")

        elif choice == '3':
            x, y, z, t = symbols("x y z t")
            variables = [x, y, z, t]
            num_eqs = 4
            equations = get_equations_from_user_nvar(num_eqs, variables)
            best_solution, best_fitness = genetic_algorithm_nvar(
                equations=equations,
                num_vars=num_eqs,
                population_size=INIT_POPULATION_SIZE_NVAR,
                generations=GENERATIONS_DEFAULT_AMOUNT_NVAR,
                value_range=(LOWER_BOUND_INIT_POP_NVAR, UPPER_BOUND_INIT_POP_NVAR),
                tournament_size=TOURNAMENT_DEFAULT_SIZE_NVAR,
                mutation_rate=MUTATION_RATE_NVAR
            )
            print(f"\nBest solution found: x={best_solution[0]:.6f}, y={best_solution[1]:.6f}, z={best_solution[2]:.6f}, t={best_solution[3]:.6f}")
            print(f"Final Fitness: {best_fitness:.8f}")
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
