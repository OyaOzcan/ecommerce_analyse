import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_data(path="ecommerce_clickstream_transactions.csv"):
    df = pd.read_csv(path)
    return df


def create_user_product_features(df):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df[df["EventType"].isin(["product_view", "add_to_cart", "purchase"])]

    user_product_stats = (
        df.groupby(["UserID", "ProductID", "EventType"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    user_product_stats.columns.name = None
    user_product_stats.rename(columns={
        "product_view": "view_count",
        "add_to_cart": "cart_count",
        "purchase": "purchase_count"
    }, inplace=True)

    for col in ["view_count", "cart_count", "purchase_count"]:
        if col not in user_product_stats.columns:
            user_product_stats[col] = 0

    user_product_stats["Label"] = (user_product_stats["purchase_count"] > 0).astype(int)
    return user_product_stats


def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df = df[df["EventType"].isin(["product_view", "add_to_cart", "purchase"])].copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def create_user_product_stats(df):
    stats = df.groupby(["UserID", "ProductID", "EventType"]).size().unstack(fill_value=0).reset_index()
    stats.rename(columns={
        "product_view": "view_count",
        "add_to_cart": "cart_count",
        "purchase": "purchase_count"
    }, inplace=True)
    for col in ["view_count", "cart_count", "purchase_count"]:
        if col not in stats.columns:
            stats[col] = 0
    stats["Label"] = (stats["purchase_count"] > 0).astype(int)
    return stats


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def segment_users(user_product_stats, n_clusters=3):
    summary = user_product_stats.groupby("UserID")[["view_count", "cart_count", "purchase_count"]].sum().reset_index()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(summary[["view_count", "cart_count", "purchase_count"]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    summary["segment"] = kmeans.fit_predict(scaled)
    return summary, scaler, kmeans


def plot_segment_behavior(user_summary):
    means = user_summary.groupby("segment")[["view_count", "cart_count", "purchase_count"]].mean()
    means.plot(kind="bar", title="Segmentlere Göre Ortalama Etkileşimler")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import random

def test_knn_loo_basic(purchases, user_item_sparse, user_encoder, product_encoder, test_limit=1000, k=5):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_sparse)
    hit_count = 0
    tested = 0
    results = []

    for user_idx in range(min(test_limit, user_item_sparse.shape[0])):
        user_products = purchases[purchases["UserID_enc"] == user_idx]["ProductID_enc"].tolist()
        if len(user_products) < 2:
            continue
        held_out = random.choice(user_products)
        vector = user_item_sparse[user_idx].toarray().copy()
        vector[0, held_out] = 0

        _, indices = model.kneighbors(vector, n_neighbors=k+1)
        similar_users = indices[0][1:]

        recommended = set()
        for sim_idx in similar_users:
            sim_prods = purchases[purchases["UserID_enc"] == sim_idx]["ProductID_enc"].tolist()
            recommended.update(sim_prods)

        hit = held_out in recommended
        hit_count += int(hit)
        tested += 1

        results.append({
            "user_id": user_encoder.inverse_transform([user_idx])[0],
            "held_out_product": product_encoder.inverse_transform([held_out])[0],
            "hit": hit,
            "recommended_count": len(recommended)
        })

    print(f"🎯 KNN Hit Rate: {hit_count}/{tested} = %{hit_count/tested*100:.2f}")
    return pd.DataFrame(results)


def test_knn_loo_success(purchases, user_item_sparse, user_encoder, product_encoder, test_limit=1000, k=5):
    from sklearn.neighbors import NearestNeighbors
    import random
    import pandas as pd

    hit_count = 0
    tested_count = 0
    results = []

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_item_sparse)

    for user_idx in range(min(test_limit, user_item_sparse.shape[0])):
        user_products = purchases[purchases['UserID_enc'] == user_idx]['ProductID_enc'].tolist()
        if len(user_products) < 2:
            continue

        held_out = random.choice(user_products)
        temp_vector = user_item_sparse[user_idx].toarray().copy()
        temp_vector[0, held_out] = 0

        distances, indices = knn_model.kneighbors(temp_vector, n_neighbors=k+1)
        similar_users = indices[0][1:]

        recommended_products = set()
        for sim_idx in similar_users:
            sim_products = purchases[purchases['UserID_enc'] == sim_idx]['ProductID_enc'].tolist()
            recommended_products.update(sim_products)

        hit = held_out in recommended_products
        hit_count += int(hit)
        tested_count += 1

        results.append({
            'user_id': user_encoder.inverse_transform([user_idx])[0],
            'held_out_product': product_encoder.inverse_transform([held_out])[0],
            'hit': hit,
            'recommended_count': len(recommended_products)
        })

    hit_rate = hit_count / tested_count if tested_count > 0 else 0
    print(f"🎯 KNN LOO Hit Rate: {hit_count}/{tested_count} = %{hit_rate * 100:.2f}")
    return pd.DataFrame(results)
