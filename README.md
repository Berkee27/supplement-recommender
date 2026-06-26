# Supplement Recommender with KNN

This is an educational machine learning project that demonstrates how a simple recommendation system can be built using the K-Nearest Neighbors algorithm.

> This project is for learning purposes only. It is not medical or supplement advice.

## Project Goal

The goal of this project is to recommend supplement-like product categories based on a user's fitness and lifestyle profile.

The system compares the user profile with product profiles and returns the closest matches.

## Technologies Used

- Python
- Pandas
- Scikit-Learn
- KNN / Nearest Neighbors
- StandardScaler

## Dataset Logic

Each row represents one product.

Each product has numeric features such as:

- muscle_gain
- fat_loss
- energy
- recovery
- general_health
- training_days
- budget_level
- caffeine_ok
- beginner_friendly

The model does not understand product names directly. It compares numeric profiles.

## Machine Learning Approach

This project uses KNN-style nearest neighbor search.

The user profile is converted into the same numeric format as the products.  
Then the model finds the nearest products using Euclidean distance.

## How to Run

Create a virtual environment:

```bash
py -m venv .venv
.\.venv\Scripts\activate
