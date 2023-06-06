import pygad
import gym

env = gym.make('LunarLander-v2')

def fitness_func(solution, sol_idx):
    total_reward = 0
    observation = env.reset()
    
    for step in range(1000):
        action = int(solution[step])
        observation, reward, terminated, truncated, info = env.step(action) 
        total_reward += reward
        if terminated or truncated:
            observation, info = env.reset()
    
    return total_reward


ga_instance = pygad.GA(
    num_generations=100,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=10,
    num_genes=1000,
    gene_type=int,
    init_range_low=0,
    init_range_high=3,
    parent_selection_type="sss",
    keep_parents=1,
    mutation_percent_genes=10,
    mutation_by_replacement=True,
    mutation_type="random",
    random_mutation_min_val=0,
    random_mutation_max_val=3,
)

#uruchomienie algorytmu
ga_instance.run()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))


#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()
