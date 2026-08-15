from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import save_model


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

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    # Save trained model
    save_model(model, "models/titanic_model.joblib")

    print("\nModel saved successfully to models/titanic_model.joblib")


if __name__ == "__main__":
    main()