# Genetic Algorithm Equation Solver

This repository contains a Python-based equation solver that utilizes a **Genetic Algorithm** to find approximate solutions for systems of linear and non-linear equations. It offers flexible options to solve problems with 2, 3, or 4 variables.

---

## Features

* **2-Variable Linear Systems:** Specifically designed for solving systems of two linear equations (e.g., $ax + by = c$).
* **N-Variable Non-Linear Systems:** A generalized solver for more complex non-linear equations with:
    * **3 variables** ($x, y, z$)
    * **4 variables** ($x, y, z, t$)
* **User-Friendly Interface:** An interactive menu guides you through selecting the type of problem you want to solve.
* **Genetic Algorithm Core:** Employs principles of natural selection, crossover, and mutation to iteratively converge towards optimal solutions.
* **Convergence Monitoring:** Includes mechanisms to detect when the algorithm has likely found a satisfactory solution or has converged.

---

## How it Works

The core of this solver is a **Genetic Algorithm (GA)**. Here's a simplified breakdown:

1.  **Initialization:** A population of random candidate solutions (sets of $x, y,$ etc. values) is generated.
2.  **Fitness Evaluation:** Each candidate solution is evaluated based on how well it satisfies the given equations. The "fitness" is typically a measure of the total error (the sum of squared differences from zero for each equation). Lower error means higher fitness.
3.  **Selection:** The fittest individuals are selected to become "parents" for the next generation.
4.  **Crossover (Recombination):** Parents exchange genetic material (parts of their solutions) to create new "offspring" solutions.
5.  **Mutation:** Random small changes are introduced to the offspring to maintain diversity and explore new solution spaces, preventing premature convergence.
6.  **Repeat:** Steps 2-5 are repeated for a set number of generations or until a sufficiently good solution is found.

For non-linear equations, the `sympy` library is used to parse the user-inputted equations, allowing for flexible and complex expressions.

---

## Getting Started

Follow these steps to get the solver up and running on your local machine.

### Prerequisites

You'll need Python 3 installed on your system.
You'll also need the following Python libraries:

* `numpy`
* `sympy`

You can install them using pip:

```bash
pip install numpy sympy
```

### Running the Solver

1.  **Clone the repository** (or download the `main.py` file directly):

    ```bash
    git clone https://github.com/YOUR_USERNAME/your-repository-name.git
    cd your-repository-name
    ```
    (Replace `YOUR_USERNAME` and `your-repository-name` with your actual GitHub username and repository name.)

2.  **Execute the Python script:**

    ```bash
    python main.py
    ```

3.  **Follow the on-screen prompts:**
    The program will present a menu. Choose the type of equation system you wish to solve and input the required equations or coefficients when prompted.

### Example Interactions:

**Solving 2-Variable Linear Equations:**

```
--- Equation Solver ---
1. Solve 2-variable linear equations (e.g., ax + by = c)
2. Solve 3-variable non-linear equations (e.g., x + y*z = c)
3. Solve 4-variable non-linear equations (e.g., x + y*z - t = c)
4. Exit
Enter your choice (1-4): 1

Enter coefficients for the first equation (a1 * x + b1 * y = c1):
input a1: 2
input b1: 1
input c1: 7

Enter coefficients for the second equation (a2 * x + b2 * y = c2):
input a2: 3
input b2: -1
input c2: 3

Solution found in generation 150
Approximate solution: x = 2.000000, y = 3.000000
```

**Solving 3-Variable Non-Linear Equations:**

```
--- Equation Solver ---
1. Solve 2-variable linear equations (e.g., ax + by = c)
2. Solve 3-variable non-linear equations (e.g., x + y*z = c)
3. Solve 4-variable non-linear equations (e.g., x + y*z - t = c)
4. Exit
Enter your choice (1-4): 2

Enter 3 equations with variables x, y, z:
Equation 1: x + y*z = 10
Equation 2: 2*x - y = 1
Equation 3: z + 3*x = 15

Parsing equation: x + y*z = 10
Parsing equation: 2*x - y = 1
Parsing equation: z + 3*x = 15
Generation 0: Fitness=16.732959 | Solution=[ 40.85243171 -10.42875151   8.3585093 ]
...
Generation 490: Fitness=0.000001 | Solution=[ 3.00000031  5.00000021  6.00000008]

Best Solution Found: [ 3.00000031  5.00000021  6.00000008] | Fitness: 0.00000108
Best solution found: x=3.000000, y=5.000000, z=6.000000
Final Fitness: 0.00000108
```

---

## Configuration

You can adjust various parameters for the Genetic Algorithm within the `main.py` file to fine-tune its performance. These constants are clearly defined at the top of the script:

* **Population Size:** `POP_SIZE_INIT_POP_2VAR`, `INIT_POPULATION_SIZE_NVAR`
* **Bounds for Initial Population:** `LOWER_BOUND_INIT_POP_2VAR`, `UPPER_BOUND_INIT_POP_2VAR`, `LOWER_BOUND_INIT_POP_NVAR`, `UPPER_BOUND_INIT_POP_NVAR`
* **Mutation Rate:** `MUTATION_RATE_2VAR`, `MUTATION_RATE_NVAR`
* **Number of Generations:** `GENERATIONS_DEFAULT_AMOUNT_NVAR`
* **Fitness Thresholds:** `BEST_FITNESS_MAX_ERROR_2VAR`, `BEST_FITNESS_MAX_ERROR_NVAR`
* **Convergence Parameters (N-Var):** `CONVERGENCE_THRESHOLD_NVAR`, `CONVERGENCE_PATIENCE_NVAR`

Experimenting with these values can significantly impact the algorithm's speed and accuracy for different problem sets.

---

## Contributing

Feel free to fork this repository, open issues, or submit pull requests if you have suggestions for improvements or new features!

---
