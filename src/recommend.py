from pathlib import Path

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "supplements.csv"

FEATURE_COLUMNS = [
    "muscle_gain",
    "fat_loss",
    "energy",
    "recovery",
    "general_health",
    "training_days",
    "budget_level",
    "caffeine_ok",
    "beginner_friendly",
]


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df


def build_knn_model(df):
    features = df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = NearestNeighbors(n_neighbors=5, metric="euclidean")
    model.fit(scaled_features)

    return model, scaler


def recommend_supplements(user_profile, df, model, scaler):
    user_df = pd.DataFrame([user_profile])
    scaled_user = scaler.transform(user_df[FEATURE_COLUMNS])

    distances, indexes = model.kneighbors(scaled_user)

    recommendations = []

    for distance, index in zip(distances[0], indexes[0]):
        supplement = df.iloc[index]

        similarity_score = 1 / (1 + distance)

        recommendations.append(
            {
                "name": supplement["name"],
                "similarity_score": similarity_score,
                "distance": distance,
                "note": supplement["note"],
            }
        )

    return recommendations


def print_recommendations(recommendations):
    print("\nEducational KNN Supplement Recommender")
    print("Bu çıktı sağlık tavsiyesi değildir. ML öğrenme amaçlıdır.\n")

    for order, item in enumerate(recommendations, start=1):
        print(f"{order}. {item['name']}")
        print(f"   Benzerlik skoru: %{item['similarity_score'] * 100:.1f}")
        print(f"   Uzaklık: {item['distance']:.3f}")
        print(f"   Açıklama: {item['note']}")
        print()


def main():
    df = load_dataset()

    model, scaler = build_knn_model(df)

    user_profile = {
        "muscle_gain": 8,
        "fat_loss": 6,
        "energy": 4,
        "recovery": 7,
        "general_health": 5,
        "training_days": 5,
        "budget_level": 6,
        "caffeine_ok": 2,
        "beginner_friendly": 8,
    }

    recommendations = recommend_supplements(
        user_profile=user_profile,
        df=df,
        model=model,
        scaler=scaler,
    )

    print_recommendations(recommendations)


if __name__ == "__main__":
    main()