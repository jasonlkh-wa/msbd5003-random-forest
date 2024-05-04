import findspark

findspark.init()
from pyspark.sql import *
from pyspark.streaming import StreamingContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from joblib import load, dump
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from time import time

RANDOM_SEED = 42
random.seed(42)
np.random.seed(RANDOM_SEED)


def init_spark(app_name="HelloWorldApp", execution_mode="local[*]"):
    spark = SparkSession.builder.master(execution_mode).appName(app_name).getOrCreate()
    sc = spark.sparkContext
    return spark, sc


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
    X = np.array(load("data/all_categorical_arr.joblib")).astype(float)
    y = np.random.randint(0, 2, X.shape[0])

    print(X.shape, y.shape)
    num_rows = 100
    num_cols = X.shape[1]

    start_time = time()
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

    df = spark.createDataFrame(
        rows,
        StructType(
            [
                StructField("X", ArrayType(ArrayType(DoubleType()))),
                StructField("y", ArrayType(IntegerType())),
            ]
        ),
    )

    # # Apply the function to each row in the DataFrame
    decision_tree_models = df.rdd.map(fit_decision_tree).collect()
    print(decision_tree_models)
    # decision_tree_models.show()

    test_sample = rows[0]
    X_test = np.array(test_sample.X)
    y_test = np.array(test_sample.y)
    res_1 = decision_tree_models[0].predict(X_test)
    res_2 = decision_tree_models[1].predict(X_test)
    print("Check if res_1 != res_2, if they are different, prolly correct")
    print(res_1)
    print(res_2)
    print(f"Total time used in seconds: {time() - start_time}")
