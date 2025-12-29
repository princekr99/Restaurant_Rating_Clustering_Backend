from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)

# 🌐 Enable CORS for all domains (or you can specify exact domain)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model & scaler
model = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load dataset
df = pd.read_csv("Dataset .csv")

FEATURES = [
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes"
]

# Internal mapping
CLUSTER_RATING = {
    0: {"stars": 5, "category": "Premium & Highly Rated"},
    1: {"stars": 4, "category": "Good Value for Money"},
    2: {"stars": 3, "category": "Average Experience"},
    3: {"stars": 2, "category": "Below Average"},
}

@app.route("/")
def home():
    return jsonify({"status": "Backend is running! 🎉"})


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_input = request.json
        city = user_input.get("city", "")
        cuisine = user_input.get("cuisine", "")
        price_pref = user_input.get("price_preference", "Medium")

        # Filter based on inputs
        filtered = df[
            (df["City"].str.contains(city, case=False, na=False)) &
            (df["Cuisines"].str.contains(cuisine, case=False, na=False))
        ]

        # Price filter
        if price_pref == "Low":
            filtered = filtered[filtered["Price range"] <= 2]
        elif price_pref == "High":
            filtered = filtered[filtered["Price range"] >= 3]

        if filtered.empty:
            return jsonify({"error": "No restaurants found for this preference"}), 404

        # Predict clusters
        X = scaler.transform(filtered[FEATURES])
        filtered["cluster"] = model.predict(X)

        results = []
        for _, row in filtered.iterrows():
            info = CLUSTER_RATING[row["cluster"]]
            results.append({
                "restaurant_name": row["Restaurant Name"],
                "stars": info["stars"],
                "category": info["category"],
                "location": row["Locality"],
                "cost_for_two": f"₹{row['Average Cost for two']}",
                "message": "Recommended for family dining 🍽️"
            })

        results = sorted(results, key=lambda x: x["stars"], reverse=True)

        return jsonify(results[:10])

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

