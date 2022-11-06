"""
Scikit-learn uses the CART algorithm which creates only binary trees
and also treats all numerical features as continuous, thus the best approach 
would be to turn all categorical features into one hot encoded ones.
"""

import numpy as np
import pandas as pd

from collections import Counter
from typing import Dict, List, Tuple


class GiniRunner:
    def run(self, probabilities):
        return 1 - sum(probabilities**2)


class EntropyRunner:
    def run(self, probabilities):
        return -sum(probabilities * np.log2(probabilities))


class TreeBaseRunner:
    def __init__(self, x: pd.DataFrame, y: pd.Series, impurity_runner=GiniRunner()):
        self.x = x
        self.y = y
        self.impurity_runner = impurity_runner

    def get_stats(self) -> None:
        impurity = self.calculate_impurity(self.y)
        print(f"impurity: {impurity}")
        print(f"samples: {self.x.shape[0]}")
        print(f"value: {Counter(self.y)}")

    def run(self) -> Tuple:
        self.get_stats()

        best_feature_to_split_on, best_feature_label_to_split_on = self.get_splits()
        print(
            f"best_feature_to_split_on: {best_feature_to_split_on}, best_feature_label_to_split_on: {best_feature_label_to_split_on}"
        )
        print("\n")

        mask_best = self.x[best_feature_to_split_on] == best_feature_label_to_split_on

        return (
            (
                self.x[mask_best].reset_index(drop=True),
                self.y[mask_best].reset_index(drop=True),
            ),
            (
                self.x[~mask_best].reset_index(drop=True),
                self.y[~mask_best].reset_index(drop=True),
            ),
        )

    def get_splits(self) -> Tuple[str, str]:
        features = self.get_features()
        best_feature_to_split_on = self.get_best_feature_to_split_on(features)
        best_feature_label_to_split_on = self.get_best_feature_label_to_split_on(
            best_feature_to_split_on
        )

        return best_feature_to_split_on, best_feature_label_to_split_on

    def calculate_impurity(self, labels) -> float:
        count_labels = np.array(list(Counter(labels).values()))
        probabilities = count_labels / len(labels)
        return self.impurity_runner.run(probabilities)

    def get_features(self):
        return self.x.columns

    def get_feature_labels(self, feature):
        return self.x[feature].value_counts().index

    def get_best_feature_to_split_on(self, features: List[str]) -> str:

        best_feature = None
        min_impurity = 1.1

        for feature in features:
            feature_impurity = self.get_feature_impurity(feature)
            if feature_impurity < min_impurity:
                best_feature = feature
                min_impurity = feature_impurity
        return best_feature

    def get_feature_impurity(self, feature) -> float:
        feature_impurity = 0
        feature_labels = self.get_feature_labels(feature)

        feature_labels_probabilities = self.get_feature_labels_probabilities(feature)

        for feature_label in feature_labels:
            feature_label_impurity = self.get_feature_label_impurity(
                feature, feature_label
            )
            feature_impurity += (
                feature_labels_probabilities.get(feature_label) * feature_label_impurity
            )
        return feature_impurity

    def get_feature_labels_probabilities(self, feature) -> Dict:
        feature_labels_probabilities = (
            self.x[feature].value_counts(normalize=True).to_dict()
        )
        return feature_labels_probabilities

    def get_feature_label_impurity(self, feature, feature_label) -> float:
        x_with_feature_label_only = self.x[feature] == feature_label
        target_labels_with_feature_label_only = self.y[x_with_feature_label_only]

        feature_label_impurity = self.calculate_impurity(
            target_labels_with_feature_label_only
        )

        return feature_label_impurity

    def get_best_feature_label_to_split_on(self, feature) -> str:
        feature_labels = self.get_feature_labels(feature)

        best_feature_label = None
        min_impurity = 1.1

        for feature_label in feature_labels:
            feature_label_impurity = self.get_feature_label_impurity(
                feature, feature_label
            )

            if feature_label_impurity < min_impurity:
                best_feature_label = feature_label
                min_impurity = feature_label_impurity
        return best_feature_label
