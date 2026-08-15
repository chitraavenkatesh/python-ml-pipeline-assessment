from src.data_loader import load_data
from src.preprocessing import preprocess_data


def main():
    # Load dataset
    df = load_data("data/titanic.csv")

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    # Preprocess dataset
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\nData preprocessing completed successfully.")
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Testing target shape: {y_test.shape}")


if __name__ == "__main__":
    main()