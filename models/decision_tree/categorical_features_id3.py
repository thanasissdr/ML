from collections import Counter
from typing import Dict, List, Protocol

import numpy as np
import pandas as pd


class ImpurityRunner(Protocol):
    def run(self, class_probabilities: np.array):
        ...


class GiniRunner(ImpurityRunner):
    def run(self, class_probabilities):
        return 1 - sum(class_probabilities**2)


class EntropyRunner(ImpurityRunner):
    def run(self, class_probabilities):
        return -sum(class_probabilities * np.log2(class_probabilities))


class TreeBaseRunner:
    def __init__(
        self, x: pd.DataFrame, y, impurity_runner: ImpurityRunner = GiniRunner()
    ):
        self.x = x
        self.y = y
        self.impurity_runner = impurity_runner

    def run(self) -> Dict | None:
        impurity = self.get_stats()

        best_feature_to_split_on = self.get_splits()
        print(f"best_feature_to_split_on: {best_feature_to_split_on}")

        if impurity == 0:

            return None

        print("\n")
        feature_labels = self.get_feature_labels(best_feature_to_split_on)

        res = {}
        res[best_feature_to_split_on] = {}
        for feature_label in feature_labels:
            mask = self.x[best_feature_to_split_on] == feature_label
            x = self.x[mask].drop([best_feature_to_split_on], axis=1)
            y = self.y[mask]

            res[best_feature_to_split_on][feature_label] = (x, y)
        return res

    def get_stats(self) -> None:
        impurity = self.calculate_impurity(self.y)
        print(f"impurity: {impurity}")
        print(f"samples: {self.x.shape[0]}")
        print(f"value: {Counter(self.y)}")

        return impurity

    def get_splits(self) -> str:
        features = self.get_features()
        best_feature_to_split_on = self.get_best_feature_to_split_on(features)

        return best_feature_to_split_on

    def calculate_impurity(self, labels) -> float:
        count_labels = np.bincount(labels)
        probabilities = count_labels / len(labels)
        return self.impurity_runner.run(probabilities)

    def get_features(self):
        return self.x.columns

    def get_feature_labels(self, feature):
        return self.x[feature].value_counts().index

    def get_best_feature_to_split_on(self, features: List[str]) -> str:
        best_feature = None
        min_impurity = (
            np.log2(self.y.nunique())
            if isinstance(self.impurity_runner, EntropyRunner)
            else 1
        )

        for feature in features:
            feature_impurity = self.get_feature_impurity(feature)
            if feature_impurity < min_impurity:
                best_feature = feature
                min_impurity = feature_impurity
        return best_feature

    def get_feature_impurity(self, feature):
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

    def get_feature_label_impurity(self, feature, feature_label):
        x_with_feature_label_only = self.x[feature] == feature_label
        target_labels_with_feature_label_only = self.y[x_with_feature_label_only]

        feature_label_impurity = self.calculate_impurity(
            target_labels_with_feature_label_only
        )

        return feature_label_impurity
