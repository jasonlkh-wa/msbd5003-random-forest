import numpy as np
from joblib import dump

if __name__ == "__main__":
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    all_categorical_arr = np.random.randint(0, 3, (10_000, 10))
    all_numerical_arr = np.random.rand(1_000, 10)

    balanced_categorical_arr = np.random.randint(0, 3, (100_000, 10))
    balanced_numerical_arr = np.random.rand(100_000, 10)
    balanced_arr = np.concatenate(
        [balanced_categorical_arr, balanced_numerical_arr], axis=1
    )

    print(f"all_categorical_arr: {all_categorical_arr.shape=}")
    print(f"all_numerical_arr: {all_numerical_arr.shape=}")
    print(f"balanced_arr: {balanced_arr.shape=}")

    dump(all_categorical_arr, "data/all_categorical_arr.joblib")
    dump(all_numerical_arr, "data/all_numerical_arr.joblib")
    dump(balanced_arr, "data/balanced_arr.joblib")
