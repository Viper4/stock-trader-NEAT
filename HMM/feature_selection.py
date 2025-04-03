import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class RandomForestFeatureSelection(object):
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.feature_importances_ = None

    def augment_data(self, bars, features):
        X = bars[features]
        y = bars["regime"].map({"Bull": 1, "Bear": -1, "Choppy": 0})  # Encode regime labels

        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def evaluate(self, bars, features):
        X_train, X_test, y_train, y_test = self.augment_data(bars, features)
        self.classifier.fit(X_train, y_train)
        self.feature_importances_ = self.classifier.feature_importances_

        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

        # Step 5: Extract Feature Importances
        feature_importances = pd.DataFrame({"Features": features, "Importance": self.classifier.feature_importances_})
        feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

        print("\nFeature Importance Ranking:")
        print(feature_importances)

        # Step 6: Select Best Features (e.g., Top 5 Features)
        best_features = feature_importances["Features"].head(5).tolist()
        print("\nSelected Features:", best_features)

        # Step 7: Re-train HMM with Best Features
        return bars[best_features]
