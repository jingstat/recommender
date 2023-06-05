import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.layers import Input, Embedding, Reshape, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
class Recommender:
    def __init__(self, n_users, n_items, n_factors=100, learning_rate=0.001):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.model = self._create_model()
        self.user_to_index_mapping = {}
        self.item_to_index_mapping = {}

    def _create_model(self):
        user = Input(shape=(1,), name="User")
        u = Embedding(self.n_users, self.n_factors, name="User-Embedding", embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(user)
        u = Reshape((self.n_factors,))(u)
        item = Input(shape=(1,), name="Item")
        m = Embedding(self.n_items, self.n_factors, name="Item-Embedding", embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(item)
        m = Reshape((self.n_factors,))(m)

        x = Dot(axes=1)([u, m])

        model = Model(inputs=[user, item], outputs=x)

        return model

    def generate_negative_samples(self, interactions, n_negative):
        negative_samples = []
        for user_id in range(self.n_users):
            user_interactions = set(interactions[interactions[:, 0] == user_id, 1])
            negative_samples.extend([(user_id, item_id) for item_id in np.random.choice(self.n_items, n_negative)
                                     if item_id not in user_interactions])
        return np.array(negative_samples)

    def fit(self, interactions, epochs=3, verbose=1, learning_rate=None):
        unique_user_ids = np.unique(interactions[:, 0])
        unique_item_ids = np.unique(interactions[:, 1])

        self.user_to_index_mapping = {user_id: index for index, user_id in enumerate(unique_user_ids)}
        self.item_to_index_mapping = {item_id: index for index, item_id in enumerate(unique_item_ids)}

        mapped_user_ids = np.array([self.user_to_index_mapping[user_id] for user_id in interactions[:, 0]])
        mapped_item_ids = np.array([self.item_to_index_mapping[item_id] for item_id in interactions[:, 1]])

        # Use Adam optimizer with custom learning rate
        if learning_rate is not None:
            self.learning_rate = learning_rate
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        self.model.fit([mapped_user_ids, mapped_item_ids], interactions[:, 2], epochs=epochs, verbose=verbose)

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = keras.models.load_model(filename)

    def get_embeddings(self):
        user_embeddings = self.model.get_layer('User-Embedding').get_weights()[0]
        item_embeddings = self.model.get_layer('Item-Embedding').get_weights()[0]
        return user_embeddings, item_embeddings

    def recommend_item_knn(self, item_id, item_embeddings, n_recommendations=3):
        neighbors = NearestNeighbors(n_neighbors=n_recommendations, metric='cosine')
        neighbors.fit(item_embeddings)
        item_index = self.item_to_index_mapping[item_id]
        _, indices = neighbors.kneighbors(item_embeddings[item_index].reshape(1, -1))
        return indices.flatten()

    def recommend(self, user_id, user_embeddings, item_embeddings, n_recommendations=3):
        neighbors = NearestNeighbors(n_neighbors=n_recommendations, metric='cosine')
        neighbors.fit(item_embeddings)
        user_index = self.user_to_index_mapping[user_id]
        user_embedding = user_embeddings[user_index]
        _, indices = neighbors.kneighbors([user_embedding])
        return indices.flatten()

    def recommend_items_to_user(self, user_id, user_embeddings, item_embeddings, interactions, n_recommendations=3):
        user_interactions = set(interactions[interactions[:, 0] == user_id, 1])
        all_items = set(range(self.n_items))
        non_interacted_items = list(all_items - user_interactions)
        item_embeddings_non_interacted = item_embeddings[non_interacted_items]

        user_index = self.user_to_index_mapping[user_id]

        indices = self.recommend(user_index, user_embeddings, item_embeddings_non_interacted, n_recommendations)
        recommended_items = [non_interacted_items[i] for i in indices]
        return recommended_items

    def recommend_users_to_item(self, item_id, user_embeddings, item_embeddings, n_recommendations=3):
        neighbors = NearestNeighbors(n_neighbors=n_recommendations, metric='cosine')
        neighbors.fit(user_embeddings)
        item_index = self.item_to_index_mapping[item_id]
        item_embedding = item_embeddings[item_index]

        _, indices = neighbors.kneighbors([item_embedding])
        return indices.flatten()