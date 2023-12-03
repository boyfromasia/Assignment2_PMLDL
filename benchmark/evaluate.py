import os
from collections import defaultdict
from surprise import SVD, Dataset, Reader
from surprise.model_selection import PredefinedKFold
from surprise import accuracy

EVAL_DATA_PATH = "data/"
MODELS_PATH = "../models/"
INTERIM_PATH = "../data/interim/"
SEED = 42

sets = [("u1.base", "u1.test"),
        ("u2.base", "u2.test"),
        ("u3.base", "u3.test"),
        ("u4.base", "u4.test"),
        ("u5.base", "u5.test"),
        ("ua.base", "ua.test"),
        ("ub.base", "ub.test")]


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


if __name__ == "__main__":
    data_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    reader = Reader(line_format='user item rating', rating_scale=(1, 5))
    full_path_sets = [(os.path.join(EVAL_DATA_PATH, x[0]), os.path.join(EVAL_DATA_PATH, x[1])) for x in sets]
    data = Dataset.load_from_folds(full_path_sets, reader=reader)
    pkf = PredefinedKFold()

    algo = SVD(random_state=SEED)
    for idx, (trainset, testset) in enumerate(pkf.split(data)):
        print(f"Evaluation {sets[idx]}")
        algo.fit(trainset)

        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=3)

        accuracy.mse(predictions)
        accuracy.rmse(predictions)
        print(f"Precision at K: {round(sum(prec for prec in precisions.values()) / len(precisions), 4)}")
        print(f"Recall at K: {round(sum(rec for rec in recalls.values()) / len(recalls), 4)}")
        print()