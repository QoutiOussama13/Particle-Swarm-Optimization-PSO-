import streamlit as st
import numpy as np
from pymoo.problems import get_problem
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Import the problem classes and MultiObjectivePSO from the notebook

class RastriginProblem:
    def __init__(self, n_var=2):
        self.n_var = n_var
        self.n_obj = 1
        self.xl = np.array([-5.12] * n_var)
        self.xu = np.array([5.12] * n_var)

    def evaluate(self, x, **kwargs):
        n = x.shape[1]
        f = 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)
        return {'F': np.array([f]).T}

    def bounds(self):
        return self.xl, self.xu

    def has_constraints(self):
        return False

    def has_bounds(self):
        return True

class MultiObjectivePSO:
    def __init__(self, problem, num_particles=50, max_gen=100, w=0.5, c1=1.5, c2=1.5):
        self.problem = problem
        self.num_particles = num_particles
        self.max_gen = max_gen
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.n_var = problem.n_var
        self.n_obj = problem.n_obj
        self.lower_bound = problem.xl
        self.upper_bound = problem.xu

        self.init_particles()
        self.history = []  # Add history tracking

    def init_particles(self):
        self.positions = np.random.uniform(
            self.lower_bound, self.upper_bound,
            (self.num_particles, self.n_var)
        )
        self.velocities = np.random.uniform(
            -1, 1, (self.num_particles, self.n_var)
        )
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.inf * np.ones((self.num_particles, self.n_obj))
        self.global_best_archive = []

    def evaluate(self, positions):
        F = np.zeros((positions.shape[0], self.n_obj))
        for i, x in enumerate(positions):
            F[i, :] = self.problem.evaluate(np.array([x]))[0]
        return F

    def dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def update_archive(self, F, positions):
        for i in range(F.shape[0]):
            new_sol = (positions[i], F[i])
            self.global_best_archive = [
                sol for sol in self.global_best_archive if not self.dominates(new_sol[1], sol[1])
            ]
            if not any(self.dominates(sol[1], new_sol[1]) for sol in self.global_best_archive):
                self.global_best_archive.append(new_sol)

    def optimize(self):
        best_scores_history = []  # Track best scores for each generation
        
        for gen in range(self.max_gen):
            scores = self.evaluate(self.positions)
            
            # Store current generation state
            gen_state = {
                'generation': gen,
                'best_positions': self.positions[np.argmin(scores[:, 0])].copy(),
                'best_score': np.min(scores),
                'mean_score': np.mean(scores),
                'archive_size': len(self.global_best_archive)
            }
            self.history.append(gen_state)

            for i in range(self.num_particles):
                if self.dominates(scores[i], self.personal_best_scores[i]):
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

            self.update_archive(scores, self.positions)

            global_best_positions = np.array([sol[0] for sol in self.global_best_archive])
            global_best = global_best_positions[np.random.choice(len(global_best_positions))]

            r1 = np.random.random((self.num_particles, self.n_var))
            r2 = np.random.random((self.num_particles, self.n_var))
            self.velocities = (
                self.w * self.velocities +
                self.c1 * r1 * (self.personal_best_positions - self.positions) +
                self.c2 * r2 * (global_best - self.positions)
            )
            self.positions += self.velocities

            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            print(f"Generation {gen + 1}/{self.max_gen}: Archive size = {len(self.global_best_archive)}")

        return self.global_best_archive, self.history

def plot_rastrigin_results(result):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = 10 * 2 + X**2 - 10 * np.cos(2 * np.pi * X) + Y**2 - 10 * np.cos(2 * np.pi * Y)
    
    # Create contour plot
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax)
    
    # Plot the best solution
    ax.scatter(result.X[0], result.X[1], color='red', s=100, label='Best Solution')
    
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("Rastrigin Optimization Results")
    ax.legend()
    
    return fig

def plot_optimization_history(history, problem_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = [h['generation'] for h in history]
    if problem_type == "single":
        best_scores = [h['best_score'] for h in history]
        mean_scores = [h['mean_score'] for h in history]
        
        ax.plot(generations, best_scores, 'b-', label='Best Score')
        ax.plot(generations, mean_scores, 'r--', label='Mean Score')
        ax.set_ylabel('Objective Value')
    
    ax.set_xlabel('Generation')
    ax.set_title('Optimization Progress')
    ax.legend()
    return fig

def create_surface_plot(solutions=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = 10 * 2 + X**2 - 10 * np.cos(2 * np.pi * X) + Y**2 - 10 * np.cos(2 * np.pi * Y)
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    
    if solutions is not None:
        # Plot current population
        objective_values = [10 * 2 + x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + 
                          x[1]**2 - 10 * np.cos(2 * np.pi * x[1]) for x in solutions]
        ax.scatter(solutions[:, 0], solutions[:, 1], objective_values,
                  color='red', alpha=0.5, s=100)
    
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("f(X1, X2)")
    return fig

def main():
    st.title("Particle Swarm Optimization Demo")
    
    problem_type = st.sidebar.selectbox(
        "Select Problem Type",
        ["Rastrigin (Single-Objective)", "ZDT1 (Multi-Objective)"]
    )
    
    if problem_type == "Rastrigin (Single-Objective)":
        st.header("Rastrigin Problem Optimization")
        
        pop_size = st.sidebar.slider("Population Size", 10, 100, 50)
        n_gen = st.sidebar.slider("Number of Generations", 10, 200, 15)
        w = st.sidebar.slider("Inertia Weight (w)", 0.1, 1.0, 0.5)
        c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 0.1, 2.0, 1.5)
        c2 = st.sidebar.slider("Social Coefficient (c2)", 0.1, 2.0, 2.0)
        
        if st.button("Run Optimization"):
            animation_container = st.empty()
            progress_container = st.empty()
            results_container = st.container()
            
            with st.spinner("Optimizing..."):
                Rastrigin = RastriginProblem(n_var=2)
                Algorithm = PSO(pop_size=pop_size, w=w, c1=c1, c2=c2)
                
                history = []
                
                def callback(algorithm):
                    # Update animation
                    current_pop = algorithm.pop.get("X")
                    fig = create_surface_plot(current_pop)
                    animation_container.pyplot(fig)
                    plt.close(fig)
                    
                    # Store history
                    history.append({
                        'generation': algorithm.n_gen,
                        'best_score': algorithm.pop.get("F").min(),
                        'mean_score': algorithm.pop.get("F").mean()
                    })
                    
                    # Update progress
                    progress_text = f"Generation {algorithm.n_gen}"
                    progress_container.text(progress_text)
                    
                    # Add small delay for visualization
                    time.sleep(0.1)
                
                result = minimize(Rastrigin,
                                Algorithm,
                                termination=('n_gen', n_gen),
                                seed=1,
                                callback=callback,
                                verbose=False)
                
                with results_container:
                    st.success("Optimization completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Best Solution:", result.X)
                    with col2:
                        st.write("Objective Value:", result.F[0])
                    
                    # Show optimization progress
                    progress_fig = plot_optimization_history(history, "single")
                    st.pyplot(progress_fig)
                    
                    # Show final solution visualization
                    final_fig = plot_rastrigin_results(result)
                    st.pyplot(final_fig)
    
    else:  # ZDT1
        st.header("ZDT1 Multi-Objective Optimization")
        
        pop_size = st.sidebar.slider("Population Size", 10, 100, 50)
        max_gen = st.sidebar.slider("Maximum Generations", 10, 200, 100)
        w = st.sidebar.slider("Inertia Weight (w)", 0.1, 1.0, 0.5)
        c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 0.1, 2.0, 1.5)
        c2 = st.sidebar.slider("Social Coefficient (c2)", 0.1, 2.0, 1.5)
        
        if st.button("Run Optimization"):
            with st.spinner("Optimizing..."):
                ZDT1 = get_problem("zdt1", n_var=20)
                pso = MultiObjectivePSO(ZDT1, num_particles=pop_size, 
                                      max_gen=max_gen, w=w, c1=c1, c2=c2)
                pareto_archive, history = pso.optimize()
                
                st.success("Optimization completed!")
                

                
                # Show Pareto front
                pareto_objectives = np.array([obj for _, obj in pareto_archive])
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], 
                          color="red", marker="x")
                ax.set_xlabel("F1")
                ax.set_ylabel("F2")
                ax.set_title("Pareto Front")
                st.pyplot(fig)

if __name__ == "__main__":
    main()
