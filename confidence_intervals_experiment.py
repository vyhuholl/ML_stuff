import argparse
from typing import Tuple

import numpy as np
import scipy.stats
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from tqdm import trange


def get_data(random_seed: int) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ]:
    """
    Generate a synthetic dataset consisting of 10 million and 2 thousand
    data points for classification. The first 1000 data points are used
    for training, the second 1000 data points are used for testing, and
    the remaining 10 000 000 data points represent the dataset
    we use to calculate the model's true performance.

    Parameters:
        random_seed: int

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray]
    """
    X, y = make_classification(
        n_samples=10_002_000,
        n_features=5,
        n_redundant=2,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=random_seed,
        flip_y=0.25,
    )

    X_train = X[:1_000]
    y_train = y[:1_000]

    X_test = X[1_000:2_000]
    y_test = y[1_000:2_000]

    X_huge_test = X[2_000:]
    y_huge_test = y[2_000:]

    return X_train, y_train, X_test, y_test, X_huge_test, y_huge_test


def normal_approx(
    clf: DecisionTreeClassifier, X_test: np.ndarray, y_test: np.ndarray
        ) -> Tuple[float, float]:
    """
    Compute normal approximation interval on a test set.

    Parameters:
        clf: DecisionTreeClassifier
        X_test: np.ndarray
        y_test: np.ndarray

    Returns:
        Tuple[float, float]
    """
    confidence = 0.95
    z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    acc_test = clf.score(X_test, y_test)

    ci_length = z_value * np.sqrt(
        (acc_test * (1 - acc_test)) / y_test.shape[0]
        )

    return acc_test - ci_length, acc_test + ci_length


def bootstrap_t(
    clf: DecisionTreeClassifier, X_train: np.ndarray, y_train: np.ndarray
        ) -> Tuple[float, float]:
    """
    Compute t confidence interval on a bootstrapped train set.

    Parameters:
        clf: DecisionTreeClassifier
        X_train: np.ndarray
        y_train: np.ndarray

    Returns:
        Tuple[float, float]
    """
    rng = np.random.RandomState(seed=42)
    idx = np.arange(y_train.shape[0])

    bootstrap_train_accuracies = []
    bootstrap_rounds = 200

    for i in range(bootstrap_rounds):
        train_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        valid_idx = np.setdiff1d(idx, train_idx, assume_unique=False)

        boot_train_X, boot_train_y = X_train[train_idx], y_train[train_idx]
        boot_valid_X, boot_valid_y = X_train[valid_idx], y_train[valid_idx]

        clf.fit(boot_train_X, boot_train_y)
        acc = clf.score(boot_valid_X, boot_valid_y)
        bootstrap_train_accuracies.append(acc)

    bootstrap_train_mean = np.mean(bootstrap_train_accuracies)

    confidence = 0.95

    t_value = scipy.stats.t.ppf(
        (1 + confidence) / 2.0, df=bootstrap_rounds - 1
        )

    se = 0.0

    for acc in bootstrap_train_accuracies:
        se += (acc - bootstrap_train_mean) ** 2

    se = np.sqrt((1.0 / (bootstrap_rounds - 1)) * se)

    ci_length = t_value * se

    return bootstrap_train_mean - ci_length, bootstrap_train_mean + ci_length


def bootstrap_percentile(
    clf: DecisionTreeClassifier, X_train: np.ndarray, y_train: np.ndarray
        ) -> Tuple[float, float]:
    """
    Compute a confidence interval on a bootstrapped train set
    using the percentile method.

    Parameters:
        clf: DecisionTreeClassifier
        X_train: np.ndarray
        y_train: np.ndarray

    Returns:
        Tuple[float, float]
    """
    rng = np.random.RandomState(seed=42)
    idx = np.arange(y_train.shape[0])

    bootstrap_train_accuracies = []
    bootstrap_rounds = 200

    for i in range(bootstrap_rounds):
        train_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        valid_idx = np.setdiff1d(idx, train_idx, assume_unique=False)

        boot_train_X, boot_train_y = X_train[train_idx], y_train[train_idx]
        boot_valid_X, boot_valid_y = X_train[valid_idx], y_train[valid_idx]

        clf.fit(boot_train_X, boot_train_y)
        acc = clf.score(boot_valid_X, boot_valid_y)
        bootstrap_train_accuracies.append(acc)

    ci_lower = np.percentile(bootstrap_train_accuracies, 2.5)
    ci_upper = np.percentile(bootstrap_train_accuracies, 97.5)

    return ci_lower, ci_upper


def bootstrap_632(
    clf: DecisionTreeClassifier, X_train: np.ndarray, y_train: np.ndarray
        ) -> Tuple[float, float]:
    """
    Compute a confidence interval on a bootstrapped train set
    using the .632 method.

    Parameters:
        clf: DecisionTreeClassifier
        X_train: np.ndarray
        y_train: np.ndarray

    Returns:
        Tuple[float, float]
    """
    rng = np.random.RandomState(seed=42)
    idx = np.arange(y_train.shape[0])

    bootstrap_train_accuracies = []
    bootstrap_rounds = 200
    weight = 0.632

    for i in range(bootstrap_rounds):
        train_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        valid_idx = np.setdiff1d(idx, train_idx, assume_unique=False)

        boot_train_X, boot_train_y = X_train[train_idx], y_train[train_idx]
        boot_valid_X, boot_valid_y = X_train[valid_idx], y_train[valid_idx]

        clf.fit(boot_train_X, boot_train_y)
        train_acc = clf.score(X_train, y_train)
        valid_acc = clf.score(boot_valid_X, boot_valid_y)
        acc = weight * train_acc + (1.0 - weight) * valid_acc

        bootstrap_train_accuracies.append(acc)

    ci_lower = np.percentile(bootstrap_train_accuracies, 2.5)
    ci_upper = np.percentile(bootstrap_train_accuracies, 97.5)

    return ci_lower, ci_upper


def bootstrap_test(
    clf: DecisionTreeClassifier, X_test: np.ndarray, y_test: np.ndarray
        ) -> Tuple[float, float]:
    """
    Compute a confidence interval on a bootstrapped test set.

    Parameters:
        clf: DecisionTreeClassifier
        X_test: np.ndarray
        y_test: np.ndarray

    Returns:
        Tuple[float, float]
    """
    predictions_test = clf.predict(X_test)

    rng = np.random.RandomState(seed=42)
    idx = np.arange(y_test.shape[0])

    test_accuracies = []
    bootstrap_rounds = 200

    for i in range(bootstrap_rounds):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        acc_test_boot = np.mean(
            predictions_test[pred_idx] == y_test[pred_idx]
            )
        test_accuracies.append(acc_test_boot)

    ci_lower = np.percentile(test_accuracies, 2.5)
    ci_upper = np.percentile(test_accuracies, 97.5)

    return ci_lower, ci_upper


def main(method: str, repetitions: int) -> None:
    """
    Test a confidence interval creating method on a synthetic dataset,
    repeated many times with different random seeds for dataset generation.
    Count and print the number of times 95% CI contains the true accuracy.

    Parameters:
        method: str
        repetitions: int

    Returns:
        None
    """
    is_inside_list = []

    for i in trange(repetitions):
        X_train, y_train, X_test, y_test, X_huge_test, y_huge_test = get_data(
            random_seed=i
            )
        clf = DecisionTreeClassifier(random_state=42, max_depth=3)
        clf.fit(X_train, y_train)
        acc_test_true = clf.score(X_huge_test, y_huge_test)
        if method == "normal_approx":
            ci_lower, ci_upper = normal_approx(clf, X_test, y_test)
        elif method == "bootstrap_t":
            ci_lower, ci_upper = bootstrap_t(clf, X_train, y_train)
        elif method == "bootstrap_percentile":
            ci_lower, ci_upper = bootstrap_percentile(clf, X_train, y_train)
        elif method == "bootstrap_632":
            ci_lower, ci_upper = bootstrap_632(clf, X_train, y_train)
        elif method == "bootstrap_test":
            ci_lower, ci_upper = bootstrap_test(clf, X_test, y_test)
        is_inside = acc_test_true >= ci_lower and acc_test_true <= ci_upper
        is_inside_list.append(is_inside)

    print(
        f"{np.mean(is_inside_list) * 100}% of 95% confidence "
        "intervals contain the true accuracy."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="confidence_intervals_experiment",
        description="""
        Test a confidence interval creating method on a synthetic dataset,
        repeated many times with different
        random seeds for dataset generation.
        Count and print the number of times 95% CI contains the true accuracy.
        """
    )

    parser.add_argument(
        "method",
        choices=[
            "normal_approx",
            "bootstrap_t",
            "bootstrap_percentile",
            "bootstrap_632",
            "bootstrap_test"
            ],
        help="confidence intervals creation method"
        )

    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        required=True,
        help="number of times to repeat"
        )

    args = parser.parse_args()
    main(args.method, args.repetitions)
