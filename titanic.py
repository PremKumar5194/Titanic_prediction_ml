import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler



class Titanic:
    def __init__(self,file="h.csv"):
        self.file = file

    def load_data(self):
        return pd.read_csv(self.file)

    def eda(self, df):
        print(df.head())
        df.info()
        print(df.describe())
        print(df.isnull().sum())
        print(df["Survived"].value_counts(normalize=True))
        print(df.groupby("Pclass")["Survived"].value_counts(normalize=True))
        print(df.groupby("Sex")["Survived"].value_counts(normalize=True))

    def handle_missing_values(self, df):
        df = df.copy()
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
        return df

    def feature_engineering(self, df):
        df = df.copy()
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss").replace("Mme", "Mrs")
        counts = df["Title"].value_counts()
        df["Title"] = df["Title"].replace(counts[counts < 10].index, "Rare")

        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[0, 12, 18, 60, 100],
            labels=["Child", "Teen", "Adult", "Senior"],include_lowest=True
        )
        df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
        df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]
        df["FareBin"] = pd.qcut(
            df["Fare"],
            4,
            labels=["Low", "Mid", "High", "VeryHigh"],
            duplicates="drop",
        )

        print(df.groupby("Title", observed=False)["Survived"].mean())
        print(df.groupby("AgeGroup", observed=False)["Survived"].mean())
        print(df.groupby("TicketGroupSize")["Survived"].mean())
        print(df.groupby("FareBin", observed=False)["Survived"].mean())
        return df

    def encode_features(self, df):
        df = df.copy()
        encoder = OrdinalEncoder(
            categories=[
                ["Child", "Teen", "Adult", "Senior"],
                ["Low", "Mid", "High", "VeryHigh"],
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        columns = ["AgeGroup", "FareBin"]
        df[columns] = encoder.fit_transform(df[columns])
        df = pd.get_dummies(
            df, columns=["Sex", "Embarked", "Title"], drop_first=True, dtype=int
        )
        return df

    def split_and_scale_data(self, df):
        X = df.drop(
            columns=["Survived", "Name", "Ticket", "Cabin", "PassengerId"],
            errors="ignore",
        )
        y = df["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
    def train_model(self, X_train, y_train):
        rf = RandomForestClassifier(
            random_state=42, max_depth=4, min_samples_leaf=5
        )
        rf.fit(X_train, y_train)

        logistic = LogisticRegression(max_iter=1000, random_state=42)
        logistic.fit(X_train, y_train)

        gradient_boosting = GradientBoostingClassifier(random_state=42)
        gradient_boosting.fit(X_train, y_train)
        return logistic, rf, gradient_boosting

    def cross_validation(self, model, X_train, y_train):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "roc_auc", "f1"],
            return_train_score=True,
        )

    def tune_model(self, model, X_train, y_train):
        if isinstance(model, RandomForestClassifier):
            grid_params = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
            }
        elif isinstance(model, GradientBoostingClassifier):
            grid_params = {
                "n_estimators": [100],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.1, 0.5],
            }
        elif isinstance(model, LogisticRegression):
            grid_params = {"C": [0.1, 1.0, 10.0]}
        else:
            raise TypeError(f"Unsupported model type: {type(model).__name__}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            cv=cv,
            scoring="accuracy",
            return_train_score=True,
        )
        grid.fit(X_train, y_train)
        print("Best Parameters:", grid.best_params_)
        print("Best Cross-Validation Score:", grid.best_score_)
        return grid

    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        results = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "classification_report": classification_report(
                y_test, predictions, zero_division=0
            ),
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }
        print("Accuracy:", results["accuracy"])
        print("F1:", results["f1"])
        print("ROC AUC:", results["roc_auc"])
        print("Confusion matrix:\n", results["confusion_matrix"])
        print("Classification report:\n", results["classification_report"])
        return results

    def run(self):
        df = self.load_data()
        self.eda(df)
        df = self.handle_missing_values(df)
        df = self.feature_engineering(df)
        df = self.encode_features(df)
        X_train, X_test, y_train, y_test = self.split_and_scale_data(df)
        models = self.train_model(X_train, y_train)
        for model in models:
            print(f"\n{type(model).__name__} grid")
            grid=self.tune_model(model, X_train, y_train)
            print(f"\n{type(model).__name__} Best Estimator:")
            print(grid.best_estimator_)
            print(f"\n{type(model).__name__} Evaluation: Using Untuned model")
            self.evaluate_model(model, X_test, y_test)
            print(f"\n{type(model).__name__} Evaluation: Using Best Estimator on Test Set")
            self.evaluate_model(grid.best_estimator_, X_test, y_test)
            print(f"{type(model).__name__} Cross-Validation:")
            scores = self.cross_validation(grid.best_estimator_, X_train, y_train)
            for metric in ("accuracy", "roc_auc", "f1"):
                print(
                    f"{metric}: "
                    f"{scores[f'test_{metric}'].mean():.4f} "
                    f"(train: {scores[f'train_{metric}'].mean():.4f})"
                )


if __name__ == "__main__":
    Titanic().run()
