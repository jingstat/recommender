import unittest
import numpy as np
from ..src.recommender import Recommender
from ..src.evaluator import Evaluator

class TestRecommender(unittest.TestCase):
    def setUp(self):
        self.recommender = Recommender(n_users=5, n_items=5)

    def test_create_model(self):
        self.assertIsNotNone(self.recommender._create_model())

    # def test_fit(self):
    #     # Here you should create some mock interactions data
    #     import numpy as np
    #     # Mock interactions
    #     n_interactions = 20
    #     mock_interactions = np.column_stack((
    #         np.random.choice(range(5), size=n_interactions),  # User IDs
    #         np.random.choice(range(5), size=n_interactions),  # Item IDs
    #         np.random.random(size=n_interactions)  # Some values
    #     ))
    #     # and use it to test the 'fit' method.
    #     self.recommender.fit(mock_interactions)
    #     # You can use assert statements to verify the behavior.
    #
    # def test_recommend_item_knn(self):
    #     # Similar to test_fit, create some mock data
    #     mock_user_embeddings, mock_item_embeddings = self.recommender.get_embeddings()
    #     recommended = self.recommender.recommend_item_knn(item_id=1, item_embeddings=mock_item_embeddings)
    #     # and use it to test the 'recommend_item_knn' method.


    # Define more test methods for the other methods in your class...



if __name__ == '__main__':
    unittest.main()
