import numpy as np
from collections import defaultdict

class Evaluator:
    def __init__(self, recommender):
        self.recommender = recommender

    def precision_at_k(self, interactions, k=10):
        user_ids = np.unique(interactions[:, 0])
        user_embeddings, item_embeddings = self.recommender.get_embeddings()

        hits = 0
        total = 0

        for user_id in user_ids:
            user_interactions = set(interactions[interactions[:, 0] == user_id, 1])
            recommended_items = self.recommender.recommend_items_to_user(
                user_id, user_embeddings, item_embeddings, interactions, k
            )

            for item_id in recommended_items:
                if item_id in user_interactions:
                    hits += 1
            total += k

        return hits / total if total > 0 else 0

    def recall_at_k(self, interactions, k=10):
        user_ids = np.unique(interactions[:, 0])
        user_embeddings, item_embeddings = self.recommender.get_embeddings()

        hits = 0
        total = 0

        for user_id in user_ids:
            user_interactions = set(interactions[interactions[:, 0] == user_id, 1])
            recommended_items = self.recommender.recommend_items_to_user(
                user_id, user_embeddings, item_embeddings, interactions, k
            )

            for item_id in recommended_items:
                if item_id in user_interactions:
                    hits += 1
            total += len(user_interactions)

        return hits / total if total > 0 else 0

    def average_precision(self, user_interactions, recommended_items):
        hits = 0
        sum_precs = 0

        for i, item_id in enumerate(recommended_items):
            if item_id in user_interactions:
                hits += 1
                sum_precs += hits / (i + 1)

        return sum_precs / len(user_interactions) if user_interactions else 0

    def mean_average_precision(self, interactions, k=10):
        user_ids = np.unique(interactions[:, 0])
        user_embeddings, item_embeddings = self.recommender.get_embeddings()

        avg_precs = []

        for user_id in user_ids:
            user_interactions = set(interactions[interactions[:, 0] == user_id, 1])
            recommended_items = self.recommender.recommend_items_to_user(
                user_id, user_embeddings, item_embeddings, interactions, k
            )

            avg_precs.append(self.average_precision(user_interactions, recommended_items))

        return np.mean(avg_precs) if avg_precs else 0

    def dcg_at_k(self, r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0
#
    def ndcg_at_k(self, interactions, k=10):
        user_ids = np.unique(interactions[:, 0])
        user_embeddings, item_embeddings = self.recommender.get_embeddings()

        ndcgs = []

        for user_id in user_ids:
            user_interactions = set(interactions[interactions[:, 0] == user_id, 1])
            recommended_items = self.recommender.recommend_items_to_user(
                user_id, user_embeddings, item_embeddings, interactions, k
            )

            r = [1 if item in user_interactions else 0 for item in recommended_items]  # Relevance scores
            dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(r)])

            # Compute IDCG
            ideal_r = sorted(r, reverse=True)  # Best possible relevance scores
            idcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_r)])

            # Compute NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)

        # Return average NDCG@k
        return np.mean(ndcgs)
#