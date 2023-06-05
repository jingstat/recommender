import numpy as np
from recommender import Recommender
from evaluator import Evaluator
import mlflow


def main():
    # Define your hyperparameters
    epochs = 3
    n_users, n_items = 1000, 3000
    # Load the interactions data
    interactions = np.load('../data/interactions.npy')

    # Initialize the recommender
    recommender = Recommender(n_users, n_items)

    # Fit the recommender to the data
    recommender.fit(interactions, epochs=epochs)

    # Get the embeddings
    user_embeddings, item_embeddings = recommender.get_embeddings()

    # Evaluate the recommender
    evaluator = Evaluator(recommender)
    precision = evaluator.precision_at_k(interactions, k=10)
    # recall = evaluator.recall_at_k(interactions, k=10)
    # map_score = evaluator.mean_average_precision(interactions, k=10)
    # ndcg = evaluator.ndcg(interactions, k=10)
    #
    # print(f"Precision@10: {precision}")
    # print(f"Recall@10: {recall}")
    # print(f"MAP@10: {map_score}")
    # print(f"NDCG@10: {ndcg}")
    #
    # # Log metrics with mlflow
    mlflow.log_metrics({
        "precision_at_10": precision,
       # "recall_at_10": recall,
       # "map_at_10": map_score,
       # "ndcg_at_10": ndcg
    })

    # Save the trained model
    recommender.save_model('model.h5')

    # Log the model with mlflow
    mlflow.log_artifact('model.h5')


if __name__ == "__main__":
    main()
