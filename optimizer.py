import pygad
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

fitness_history = []

def build_model(shape, n1, n2, lr):

    from tensorflow.keras.optimizers import Adam

    model = Sequential()

    model.add(Flatten(input_shape=shape))
    model.add(Dense(n1, activation="relu"))
    model.add(Dense(n2, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr), loss="mse")

    return model


def run_ga(X_train, y_train):

    def fitness_func(ga, solution, idx):

        n1 = int(solution[0])
        n2 = int(solution[1])
        lr = solution[2]
        batch = int(solution[3])

        model = build_model(X_train.shape[1:], n1, n2, lr)

        model.fit(X_train, y_train, epochs=5, batch_size=batch, verbose=0)

        loss = model.evaluate(X_train, y_train, verbose=0)

        fitness = 1/(loss+1e-6)

        fitness_history.append(fitness)

        return fitness

    gene_space = [
        {"low":32,"high":128},
        {"low":16,"high":64},
        {"low":0.0001,"high":0.01},
        {"low":16,"high":64}
    ]

    ga = pygad.GA(
        num_generations=5,
        sol_per_pop=6,
        num_parents_mating=3,
        num_genes=4,
        gene_space=gene_space,
        fitness_func=fitness_func
    )

    ga.run()

    solution, fitness, _ = ga.best_solution()

    return solution, fitness_history