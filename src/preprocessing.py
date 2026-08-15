import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df: pd.DataFrame):
    """
    Clean and preprocess the Titanic dataset.

    Args:
        df: Raw Titanic dataset.

    Returns:
        X_train, X_test, y_train, y_test
    """

    # Create a copy to avoid modifying the original dataframe
    data = df.copy()

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Drop columns that are not useful for this model
    columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    data = data.drop(columns=columns_to_drop)

    # Fill missing Age values with the median
    data["Age"] = data["Age"].fillna(data["Age"].median())

    # Fill missing Embarked values with the most common value
    data["Embarked"] = data["Embarked"].fillna(
        data["Embarked"].mode()[0]
    )

    # Separate features and target
    X = data.drop("Survived", axis=1)
    y = data["Survived"]

    # Convert categorical columns into numeric columns
    X = pd.get_dummies(
        X,
        columns=["Sex", "Embarked"],
        drop_first=True
    )

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test