from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open(r"C:\Users\raman\OneDrive\Desktop\Restaurant_Rating_Clustering\kmeans_model.pkl", "rb"))
scaler = pickle.load(open(r"C:\Users\raman\OneDrive\Desktop\Restaurant_Rating_Clustering\scaler.pkl", "rb"))

# Load dataset
df = pd.read_csv(r"C:\Users\raman\OneDrive\Desktop\Restaurant_Rating_Clustering\Dataset .csv")

FEATURES = [
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes"
]

# Cluster → Rating Mapping (INTERNAL)
CLUSTER_RATING = {
    0: {"stars": 5, "category": "Premium & Highly Rated"},
    1: {"stars": 4, "category": "Good Value for Money"},
    2: {"stars": 3, "category": "Average Experience"},
    3: {"stars": 2, "category": "Below Average"}
}

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.json

    city = user_input["city"]
    cuisine = user_input["cuisine"]
    price_pref = user_input["price_preference"]

    # Filter restaurants based on preference
    filtered = df[
        (df["City"].str.contains(city, case=False, na=False)) &
        (df["Cuisines"].str.contains(cuisine, case=False, na=False))
    ]

    if price_pref == "Low":
        filtered = filtered[filtered["Price range"] <= 2]
    elif price_pref == "High":
        filtered = filtered[filtered["Price range"] >= 3]

    # Predict clusters
    X = scaler.transform(filtered[FEATURES])
    filtered["cluster"] = model.predict(X)

    # Convert to user-friendly output
    results = []
    for _, row in filtered.iterrows():
        rating_info = CLUSTER_RATING[row["cluster"]]

        results.append({
            "restaurant_name": row["Restaurant Name"],
            "stars": rating_info["stars"],
            "category": rating_info["category"],
            "location": row["Locality"],
            "cost_for_two": f"₹{row['Average Cost for two']}",
            "message": "Recommended for family dining 🍽️"
        })

    # Sort by stars
    results = sorted(results, key=lambda x: x["stars"], reverse=True)

    return jsonify(results[:10])  # Top 10 restaurants

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))