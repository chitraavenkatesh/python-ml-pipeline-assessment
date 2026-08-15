from src.data_loader import load_data


def main():
    df = load_data("data/titanic.csv")

    print("Dataset loaded successfully.")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")


if __name__ == "__main__":
    main()