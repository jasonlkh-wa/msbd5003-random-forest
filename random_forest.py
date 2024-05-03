# %%
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql import functions as F, SparkSession
from pyspark.sql.types import *
from joblib import load
from common_utils import printls
import argparse
import findspark
import typing as t
import random
import math as m

RANDOM_SEED = 42


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Use local spark")
    args = parser.parse_args()
    return args


def init_spark(app_name="HelloWorldApp", execution_mode="local[*]"):
    findspark.init()
    spark = t.cast(
        SparkSession,
        SparkSession.builder.master(execution_mode).appName(app_name).getOrCreate(),
    )
    sc = spark.sparkContext
    return spark, sc


def class_entropy(df):
    # Example entropy calculation for binary classification
    col_name = "y"
    counts = df.groupBy(col_name).count()
    total = df.count()
    return (
        counts.withColumn("prob", F.col("count") / total)
        .select(F.sum(-F.col("prob") * F.log2(F.col("prob"))).alias("entropy"))
        .first()["entropy"]
    )


def prob(df):
    # Example entropy calculation for binary classification
    count = np.count_nonzero(df)
    if count == 0:
        return float(0)
    else:
        total = len(df)
        prob = np.divide(count, total)
        return float(prob)


def load_training_data(dataset: str = "all_categorical"):
    if dataset == "all_categorical":
        X = np.array(load("./data/all_categorical_arr.joblib"))
        y = np.array(load("./data/y_categorical.joblib"))
    else:
        X = np.array(load("./data/all_categorical_arr.joblib"))
        y = np.array(load("./data/y_categorical.joblib"))

    printls(f"{X.shape=}{y.shape=}")
    return X, y


def new_split(joined_df, feature_index):

    # Select relevant columns
    feature_col_name = joined_df.columns[feature_index]
    y_col_name = joined_df.columns[-1]
    split_data = (
        joined_df.select(feature_col_name, y_col_name)
        .withColumnRenamed(feature_col_name, "feature")
        .withColumnRenamed(y_col_name, "y")
    )

    # Calculate parent entropy
    parent_entropy = class_entropy(joined_df.select("y"))
    parent_data_count = joined_df.count()

    # Calculate potential splits and their Information Gain
    distinct_values = (
        split_data.select("feature")
        .withColumnRenamed("feature", "split_value")
        .distinct()
        .orderBy("split_value")
    )

    # Cartesian join to get split mask
    splits_info = distinct_values.crossJoin(split_data).withColumn(
        "is_left", F.col("feature") <= F.col("split_value")
    )

    # aggregate list
    entropies = splits_info.groupBy("split_value", "is_left").agg(
        F.count("y").alias("count"),
        F.sum("y").alias("sum"),
        prob_udf(F.collect_list("y")).alias("prob"),
    )
    entropies = entropies.withColumn(
        "entropy",
        -F.col("prob") * F.log2(F.col("prob"))
        - (1 - F.col("prob")) * F.log2((1 - F.col("prob"))),
    )
    # Calculate Information Gain for each split
    info_gain = entropies.groupBy("split_value").agg(
        (
            parent_entropy
            - F.sum(F.col("entropy") * (F.col("count") / parent_data_count))
        ).alias("info_gain")
    )

    # Get the best split
    best_split = info_gain.orderBy(F.desc("info_gain")).first()

    schema = StructType(
        [
            StructField("feature", IntegerType(), True),
            StructField("split_value", FloatType(), True),
            StructField("info_gain", FloatType(), True),
        ]
    )

    # Prepare output DataFrame
    result_df = spark.createDataFrame(
        [(feature_index, float(best_split["split_value"]), best_split["info_gain"])],
        schema,
    )

    return result_df


def feature_split(dataset, feature_array):
    """
    Input:
    partition: a pyspark dataframe partition to be called by foreachPartition,
    feature_array: a broadcasted feature array for the tree that is intiialized earlier on
    """
    # define schema
    schema = StructType(
        [
            StructField("feature", IntegerType(), True),
            StructField("split_value", FloatType(), True),
            StructField("info_gain", FloatType(), True),
        ]
    )
    feature_df = spark.createDataFrame([], schema)

    # for each feature array, get a split and append the dataframe
    for feature_index in feature_array:

        # find split
        feature_split = new_split(dataset, feature_index)

        # add feature
        feature_df = feature_df.union(feature_split)

    return feature_df


def random_forest_train(df, num_trees, max_depth=3):
    trees = []
    num_features = int(len(df.columns[:-1]))

    for _ in range(num_trees):

        # sample dataset with replacement
        # to be replaced with sampling method from Jason
        sampled_df = df.sample(withReplacement=True, fraction=1.0)

        # sample features
        # to be replaced with a more updated version if available
        feature_array = random.sample(
            range(num_features), k=int(m.log(num_features, 2) + 1)
        )

        tree = grow_tree(sampled_df, feature_array, max_depth)
        trees.append(tree)

        # create node

    return trees


def grow_tree(df, feature_array, max_depth=3):

    # init
    y_label = df.columns[-1]
    node = {}
    np.zeros()

    # get first tree
    feature_df = feature_split(df, feature_array)
    feature_list = feature_df.collect()

    # init
    feature_idx = feature[0]
    best_split = feature[1]
    gain = feature[2]

    # generate split
    left_df = df.filter(col(joined_df.columns[feature_idx]) <= best_split)
    right_df = df.filter(col(joined_df.columns[feature_idx]) <= best_split)

    return (feature_df, left_df, right_df)


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    args = init_args()
    # kleung: Uncomment this when you run in spark clusters
    # if args.local:
    spark, sc = init_spark()

    X, y = load_training_data()

    X_data = spark.createDataFrame(X)
    y_data = spark.createDataFrame(y)

    X_train = X_data.withColumn("id", monotonically_increasing_id())
    y_train = y_data.withColumn("id", monotonically_increasing_id())

    # DEVELOPMENT: create joined df for computation with one id
    # kleung: feel unnecessary join here
    # joined_df = X_indexed.join(y_indexed, "id").drop("id")
    print(X_train.columns)
    print(y_train.columns)

    # Bootstrap function definition

    # Weighted bootstrap subdataset

    # partition the dataframe dataset

    # bootstrap sampling per tree
    # create variations of the joined_df baseed on bootstramp algorithm

    # boostrap data setsplit

    # Define entropy for classification evaluation crtieria
    # Receives a probably as input to calculate entropy

    class_entropy_udf = udf(class_entropy, ArrayType(DoubleType()))
    prob_udf = udf(prob, FloatType())

    # test = new_split(joined_df, 0).show()

    # Build tree from splitting

    # Each tree
    # (i) for each feature: find_split
    # (ii) Mapbypartition(find_split)
