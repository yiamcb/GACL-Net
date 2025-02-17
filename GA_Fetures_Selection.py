from deap import base, creator, tools, algorithms 
import numpy as np

# Define fitness function for GA
def evaluate_individual(individual, X_train, y_train, X_val, y_val, input_shape, num_classes):
    # Select features based on individual
    selected_features = [index for index, value in enumerate(individual) if value == 1]
    if len(selected_features) == 0:
        return 0,
    
    X_train_fs = X_train[:, selected_features]
    X_val_fs = X_val[:, selected_features]

    temp_model = build_model((len(selected_features), 1))
    temp_model.fit(X_train_fs, y_train, epochs=3, batch_size=32, verbose=0)
    
    _, accuracy = temp_model.evaluate(X_val_fs, y_val, verbose=0)
    return accuracy,

num_features = X_train.shape[1]
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_individual, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, 
                 input_shape=input_shape, num_classes=num_classes)

population = toolbox.population(n=50)
ngen = 4  
hof = tools.HallOfFame(1)  # Best individual
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, halloffame=hof, verbose=True)

best_individual = hof[0]
selected_features = [index for index, value in enumerate(best_individual) if value == 1]
print(f"Selected Features: {selected_features}")

# Subset the data with the selected features
X_train_fs = X_train[:, selected_features]
X_val_fs = X_val[:, selected_features]
X_test_fs = X_test[:, selected_features]

# Train the model with selected features
input_shape_fs = (len(selected_features), 1)
model_fs = build_model(input_shape_fs)
history_fs = model_fs.fit(
    X_train_fs, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val_fs, y_val)
)

test_loss_fs, test_accuracy_fs = model_fs.evaluate(X_test_fs, y_test)
print(f"Test Loss (Selected Features): {test_loss_fs:.4f}, Test Accuracy (Selected Features): {test_accuracy_fs:.4f}")