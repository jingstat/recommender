import numpy as np
from recommender import Recommender
# Assume we have 1000 users and 3000 items.
n_users, n_items = 1000, 3000

# And we have some interactions data as a numpy array, where
# the first column is the user ID, the second column is the item ID,
# and the third column is the rating.
# For example, we can randomly generate some data:
np.random.seed(0)  # for reproducibility
interactions = np.random.randint(0, 5, (10000, 3))

# We will map user and item IDs to start from 0 and be contiguous
interactions[:, 0] = np.random.choice(n_users, size=10000)
interactions[:, 1] = np.random.choice(n_items, size=10000)

# Initialize the recommender
recommender = Recommender(n_users, n_items)

# Generate negative samples and concatenate with interactions
negative_samples = recommender.generate_negative_samples(interactions, n_negative=10)
all_samples = np.concatenate([interactions, np.column_stack([negative_samples, np.zeros((negative_samples.shape[0],))])])

# Fit the recommender to the data
recommender.fit(all_samples)

# Save the trained model
recommender.save_model('model.h5')

# Load the model
recommender.load_model('model.h5')

# Get the embeddings
user_embeddings, item_embeddings = recommender.get_embeddings()

# Recommend 10 items similar to item 0
similar_items = recommender.recommend(item_id=0, item_embeddings=item_embeddings)
print(f"Items similar to item 0: {similar_items}")
