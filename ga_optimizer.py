import pygad
import numpy as np
from sklearn.metrics import mean_squared_error
from models import build_mlp


def run_ga(X_train, y_train, X_val, y_val):

    fitness_history = []

    def fitness_func(ga_instance, solution, solution_idx):

        n1 = int(solution[0])
        n2 = int(solution[1])
        lr = solution[2]
        batch = int(solution[3])

        model = build_mlp(X_train.shape[1:], n1, n2, lr)

        model.fit(
            X_train,
            y_train,
            epochs=6,   # slightly longer
            batch_size=batch,
            verbose=0
        )

        preds = model.predict(X_val, verbose=0)

        rmse = np.sqrt(mean_squared_error(y_val, preds))

        fitness = 1 / (rmse + 1e-6)

        fitness_history.append(fitness)

        return fitness


    gene_space = [

        {"low":32,"high":128},     # layer1 neurons
        {"low":16,"high":64},      # layer2 neurons
        {"low":0.0005,"high":0.01},# learning rate
        {"low":16,"high":64}       # batch size
    ]


    ga = pygad.GA(

        num_generations=6,
        sol_per_pop=6,
        num_parents_mating=3,
        num_genes=4,
        gene_space=gene_space,
        mutation_probability=0.2,
        mutation_num_genes=1,
        fitness_func=fitness_func

    )

    ga.run()

    solution, fitness, _ = ga.best_solution()

    return solution, fitness_history