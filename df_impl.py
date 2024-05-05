# import findspark
from pyspark.sql import *  # type: ignore
from pyspark.streaming import StreamingContext
from pyspark.sql.types import *  # type: ignore
from pyspark.sql.functions import *  # type: ignore
from joblib import load, dump
import random
import numpy as np
import typing as t

# from sklearn.tree import DecisionTreeClassifier
from time import time

RANDOM_SEED = 42
random.seed(42)
np.random.seed(RANDOM_SEED)

# findspark.init()


def printls(*s):
    s = " ".join([str(x) for x in s])
    print(f"{'-'*10}\n{s}\n{'-'*10}")


def init_spark(
    app_name="HelloWorldApp", execution_mode="local[*]"
) -> tuple[SparkSession, SparkContext]:
    spark = SparkSession.builder.master(execution_mode).appName(app_name).getOrCreate()  # type: ignore
    sc = spark.sparkContext
    return spark, sc


class DecisionTreeClassifier:
    def __init__(self, max_depth=10, criterion="entropy"):
        self.max_depth = max_depth
        self.criterion: t.Literal["gini", "entropy"] = t.cast(
            t.Literal["entropy"], criterion
        )
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return self._create_leaf_node(y)

        feature_indices = self._get_feature_indices(X.shape[1])
        best_split_feature, best_split_value = self._find_best_split(
            X, y, feature_indices
        )

        left_indices = X[:, best_split_feature] <= best_split_value
        right_indices = X[:, best_split_feature] > best_split_value

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return self._create_leaf_node(y)

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "split_feature": best_split_feature,
            "split_value": best_split_value,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _get_feature_indices(self, num_features):
        return np.arange(num_features)

    def _find_best_split(self, X, y, feature_indices):
        best_split_feature = None
        best_split_value = None
        best_score = float("inf")

        for feature in feature_indices:
            if len(np.unique(X[:, feature])) > 10:
                unique_values = np.linspace(
                    np.min(X[:, feature]), np.max(X[:, feature]), 10
                )
            else:
                unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_indices = X[:, feature] <= value
                right_indices = X[:, feature] > value

                if self.criterion == "entropy":
                    score = self._entropy(y[left_indices], y[right_indices])
                else:
                    score = self._gini_index(y[left_indices], y[right_indices])

                if score < best_score:
                    best_score = score
                    best_split_feature = feature
                    best_split_value = value

        return best_split_feature, best_split_value

    def _entropy(self, y_left, y_right):
        if len(y_left) == 0 or len(y_right) == 0:
            return 0.0

        _, counts = np.unique(y_left, return_counts=True)
        probabilities = counts / len(y_left)
        entropy_left = -np.sum(probabilities * np.log2(probabilities))

        _, counts = np.unique(y_right, return_counts=True)
        probabilities = counts / len(y_right)
        entropy_right = -np.sum(probabilities * np.log2(probabilities))

        weighted_entropy = (
            len(y_left) / (len(y_left) + len(y_right)) * entropy_left
            + len(y_right) / (len(y_left) + len(y_right)) * entropy_right
        )
        return weighted_entropy

    def _gini_index(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        gini_left = self._calculate_gini(y_left)
        gini_right = self._calculate_gini(y_right)
        weighted_gini = (len(y_left) / n) * gini_left + (len(y_right) / n) * gini_right
        return weighted_gini

    def _calculate_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _create_leaf_node(self, y):
        if len(y) == 0:
            raise ValueError("Empty array y provided.")

        _, counts = np.unique(y, return_counts=True)
        majority_class = np.argmax(counts)
        return {"class": majority_class}

    def _traverse_tree(self, x, node):
        while "class" not in node:
            if x[node["split_feature"]] <= node["split_value"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["class"]


def fit_decision_tree(row):
    # Convert the row into numpy arrays
    X = np.array(row.X)
    y = np.array(row.y)

    # Fit a decision tree model
    decision_tree = DecisionTreeClassifier()
    model = decision_tree.fit(X, y)

    return model


if __name__ == "__main__":
    spark, sc = init_spark()
    X = np.random.rand(100_000, 10)
    y = np.random.randint(0, 2, X.shape[0])
    printls(f"max parallel: {sc.defaultParallelism}")
    print(X.shape, y.shape)
    num_rows = 100
    num_cols = X.shape[1]

    rows = []
    for _ in range(num_rows):
        # Randomly select indices with replacement for bootstrapping
        indices = random.choices(range(X.shape[0]), k=int(X.shape[0] * 0.1))

        # Randomly select half of the columns for each subset
        selected_cols = random.sample(range(num_cols), num_cols // 2)

        # Create a new row with bootstrapped X and corresponding y
        new_row = Row(X=X[indices][:, selected_cols].tolist(), y=y[indices].tolist())

        # Add the row to the list
        rows.append(new_row)

    for n_partition in range(8, 129, 8):
        start_time = time()
        df = spark.createDataFrame(
            rows,
            StructType(
                [
                    StructField("X", ArrayType(ArrayType(DoubleType()))),
                    StructField("y", ArrayType(IntegerType())),
                ]
            ),
        )

        if n_partition > 0:
            df = df.repartition(n_partition)

        # Apply the function to each row in the DataFrame
        decision_tree_models = df.rdd.map(fit_decision_tree).collect()

        test_sample = rows[0]
        X_test = np.array(test_sample.X)
        y_test = np.array(test_sample.y)
        res_1 = decision_tree_models[0].predict(X_test)
        res_2 = decision_tree_models[1].predict(X_test)
        printls(f"{n_partition} -  Total time used in seconds: {time() - start_time}")
