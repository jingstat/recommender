import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, ParameterGrid
from recommender import Recommender
from evaluator import Evaluator
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Assume we have 1000 users and 3000 items.
n_users, n_items = 1000, 3000

np.random.seed(0)  # for reproducibility
interactions = np.random.randint(0, 5, (10000, 3))

interactions[:, 0] = np.random.choice(n_users, size=10000)
interactions[:, 1] = np.random.choice(n_items, size=10000)

# Initialize the recommender
recommender = Recommender(n_users, n_items)

# Generate negative samples and concatenate with interactions
negative_samples = recommender.generate_negative_samples(interactions, n_negative=10)
all_samples = np.concatenate([interactions, np.column_stack([negative_samples, np.zeros((negative_samples.shape[0],))])])

# Split data into training and test sets
train_samples, test_samples = train_test_split(all_samples, test_size=0.2, random_state=42)

# Define the objective function that the fmin will minimize
def objective(params):
    with mlflow.start_run():
        # Fit the recommender with the current hyperparameters
        recommender.fit(train_samples, epochs=int(params['epochs']), learning_rate=params['learning_rate'])

        # Calculate the metrics with the evaluator
        evaluator = Evaluator(recommender)
        precision = evaluator.precision_at_k(test_samples)
        mse = recommender.model.evaluate([test_samples[:,0],test_samples[:,1]],test_samples[:,2])
        # Log the metrics to MLFlow
        mlflow.log_param("epochs", int(params['epochs']))
        mlflow.log_param("learning_rate", params['learning_rate'])
        mlflow.log_metric("precision_at_k", precision)
        mlflow.log_metric("mse", mse)

        # We want to maximize precision, so we return a negative value
        return {'loss': mse, 'status': STATUS_OK}

# Define the hyperparameters to optimize
param_space = {
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
    'epochs': hp.quniform('epochs', 3, 5, 1)
}

# Run the optimizer
trials = Trials()
best = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=2, trials=trials)

print("Best parameters:", best)
