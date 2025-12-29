import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\raman\OneDrive\Desktop\Restaurant_Rating_Clustering\Dataset .csv")

# Select numeric features for clustering
features = [
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes"
]

X = df[features]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Elbow Method
# -----------------------------
wcss = []
K = range(1, 11)

for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.plot(K, wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# -----------------------------
# Train Final Model (K = 4)
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# -----------------------------
# Save Model & Scaler
# -----------------------------
pickle.dump(kmeans, open(r"C:\Users\raman\OneDrive\Desktop\Restaurant_Rating_Clustering\kmeans_model.pkl", "wb"))
pickle.dump(scaler, open(r"C:\Users\raman\OneDrive\Desktop\Restaurant_Rating_Clustering\scaler.pkl", "wb"))

print("✅ Model and scaler saved successfully")