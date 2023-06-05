import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, ParameterGrid
from recommender import Recommender
from evaluator import Evaluator

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
# Define the hyperparameters to test
param_grid = {'epochs': [5, 10], 'learning_rate': [0.001, 0.01, 0.1]}

# Loop over each combination of hyperparameters
for params in ParameterGrid(param_grid):
    with mlflow.start_run():
        # Fit the recommender with the current hyperparameters
        recommender.fit(train_samples, epochs=params['epochs'], learning_rate=params['learning_rate'])

        # Calculate the metrics with the evaluator
        evaluator = Evaluator(recommender)
        precision = evaluator.precision_at_k(test_samples)
        # recall = evaluator.recall_at_k(interactions)
        # map = evaluator.mean_average_precision(interactions)
        # ndcg = evaluator.ndcg_at_k(interactions)

        # Log the metrics to MLFlow
        mlflow.log_param("epochs", params['epochs'])
        mlflow.log_param("learning_rate", params['learning_rate'])
        mlflow.log_metric("precision_at_k", precision)
        # mlflow.log_metric("recall_at_k", recall)
        # mlflow.log_metric("mean_average_precision", map)
        # mlflow.log_metric("ndcg_at_k", ndcg)
