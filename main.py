from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_model


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

    # Train model
    model = train_model(X_train, y_train)

    print("\nModel training completed successfully.")


if __name__ == "__main__":
    main()