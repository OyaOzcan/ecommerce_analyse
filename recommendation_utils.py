import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori, association_rules

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st

def prepare_datasets(df):
    # 1. Sadece geçerli EventType kayıtlarını al
    df = df[df["EventType"].isin(["product_view", "add_to_cart", "purchase"])].copy()

    # 2. ProductID olmayanları çıkar
    df = df[df["ProductID"].notnull()].copy()

    # 3. Encode işlemleri
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    df["UserID_enc"] = user_encoder.fit_transform(df["UserID"])
    df["ProductID_enc"] = product_encoder.fit_transform(df["ProductID"])

    # 4. Kullanıcı x Ürün interaksiyon matrisi
    interaction_matrix = df.pivot_table(
        index="UserID_enc",
        columns="ProductID_enc",
        values="EventType",
        aggfunc="count",
        fill_value=0
    )

    # 5. Sadece satın alma işlemlerini içeren DataFrame
    purchase_df = df[df["EventType"] == "purchase"][["UserID", "ProductID"]].copy()

    # 6. Her kullanıcıdan bir ürün (LOO testi için held-out)
    user_summary = (
        df[df["EventType"] == "purchase"]
        .sort_values("Timestamp")
        .drop_duplicates(subset="UserID", keep="last")[["UserID", "ProductID"]]
        .copy()
    )

    # 7. Kullanıcı bazlı davranış özeti (view, cart, purchase sayısı)
    stats = (
        df.pivot_table(index="UserID_enc", columns="EventType", aggfunc="size", fill_value=0)
        .reset_index()
        .rename(columns={
            "product_view": "view_count",
            "add_to_cart": "cart_count",
            "purchase": "purchase_count"
        })
    )
    stats["Label"] = stats["purchase_count"].apply(lambda x: 1 if x > 0 else 0)

    return stats, interaction_matrix, purchase_df, user_summary, user_encoder, product_encoder


def get_knn_recommendations(interaction_matrix, selected_user_id, n_neighbors=5, n_recommend=5):
    knn_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    knn_model.fit(interaction_matrix)

    # 🛠️ reshape zorunlu
    distances, indices = knn_model.kneighbors(
        interaction_matrix.loc[selected_user_id].values.reshape(1, -1)
    )

    similar_users = interaction_matrix.index[indices.flatten()[1:]]  # Kendisi hariç

    user_vector = interaction_matrix.loc[selected_user_id]
    recommended_items = []

    for sim_user in similar_users:
        sim_vector = interaction_matrix.loc[sim_user]
        recommend = (sim_vector == 1) & (user_vector == 0)
        recommended_items.extend(recommend[recommend].index.tolist())

    return Counter(recommended_items).most_common(n_recommend)


# Apriori kuralları döner
def get_apriori_rules(df, min_support=0.001, metric="lift", min_threshold=0.5):
    from sklearn.preprocessing import LabelEncoder
    apriori_df = df[df["EventType"] == "purchase"].copy()
    apriori_df = apriori_df[["UserID", "ProductID"]].dropna()

    # Encoding işlemleri burada yapılmalı
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    apriori_df["UserID_enc"] = user_encoder.fit_transform(apriori_df["UserID"])
    apriori_df["ProductID_enc"] = product_encoder.fit_transform(apriori_df["ProductID"])

    apriori_df["Purchased"] = 1

    basket = apriori_df.pivot_table(index="UserID_enc", columns="ProductID_enc", values="Purchased", fill_value=0)
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        return pd.DataFrame()
    
    return association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)


# KMeans segmentleri ve önerilen ürünleri döner
def get_segment_recommendations(df, k=3):
    from sklearn.preprocessing import LabelEncoder

    # Eğer 'UserID_enc' yoksa encode et
    if 'UserID_enc' not in df.columns or 'ProductID_enc' not in df.columns:
        user_encoder = LabelEncoder()
        product_encoder = LabelEncoder()
        df["UserID_enc"] = user_encoder.fit_transform(df["UserID"])
        df["ProductID_enc"] = product_encoder.fit_transform(df["ProductID"])

    user_summary = df.pivot_table(index="UserID_enc", columns="EventType", aggfunc="size", fill_value=0).reset_index()

    # KMeans segmentleme
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    user_summary["segment"] = kmeans_model.fit_predict(user_summary[["add_to_cart", "product_view", "purchase"]])

    # Segmentlere göre öneri listesi
    segment_results = {}
    for segment_id in user_summary["segment"].unique():
        users_in_segment = user_summary[user_summary["segment"] == segment_id]["UserID_enc"]
        segment_df = df[(df["UserID_enc"].isin(users_in_segment)) & (df["EventType"] == "purchase")]
        top_products = Counter(segment_df["ProductID_enc"]).most_common(5)
        segment_results[segment_id] = top_products

    return segment_results, user_summary


# Hibrit skorlamayı hesaplar
def get_hybrid_recommendations(knn_items, apriori_items, segment_items, weights=None, top_k=10):
    if weights is None:
        weights = {"knn": 3, "apriori": 2, "segment": 1}

    # Listeleri ağırlıkla çoğalt
    all_items = (
        knn_items * weights["knn"] +
        apriori_items * weights["apriori"] +
        segment_items * weights["segment"]
    )

    hybrid_counts = Counter(all_items)
    return hybrid_counts.most_common(top_k)

def train_logistic_regression(X_train, y_train, X_test, y_test, C=1.0, max_iter=300):
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred), model

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred), model

def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'report_df': pd.DataFrame(report).transpose(),
        'confusion_matrix': cm
    }
    return metrics

def run_selected_model(X, y, model_name, test_size=0.3, random_state=42, **model_params):
    # Eğitim ve test verilerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Model seçimi
    if model_name == 'LogisticRegression':
        metrics, model = train_logistic_regression(X_train, y_train, X_test, y_test, **model_params)
    elif model_name == 'RandomForest':
        metrics, model = train_random_forest(X_train, y_train, X_test, y_test, **model_params)
    else:
        raise ValueError("Geçersiz model adı! 'LogisticRegression' veya 'RandomForest' kullanın.")

    return metrics, model

def compare_models(X, y, test_size=0.3):
    results = {}

    # Lojistik Regresyon
    log_metrics, _ = run_selected_model(X, y, model_name='LogisticRegression', test_size=test_size, C=1.0, max_iter=300)
    results['Logistic Regression'] = log_metrics

    # Random Forest
    rf_metrics, _ = run_selected_model(X, y, model_name='RandomForest', test_size=test_size, n_estimators=100, max_depth=10)
    results['Random Forest'] = rf_metrics

    # Karşılaştırma tablosu
    comparison_df = pd.DataFrame({
        model: {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score']
        }
        for model, metrics in results.items()
    }).T

    return comparison_df

def plot_segments(user_item_matrix, n_clusters=4):
    # Segmentleme
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    segments = kmeans.fit_predict(user_item_matrix)

    # PCA ile 2 boyuta indirgeme
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(user_item_matrix)

    # DataFrame'e ekle
    df_plot = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
    df_plot['Segment'] = segments

    # Plot
    plt.figure(figsize=(10, 6))
    for seg in sorted(df_plot['Segment'].unique()):
        subset = df_plot[df_plot['Segment'] == seg]
        plt.scatter(subset['PC1'], subset['PC2'], label=f'Segment {seg}', alpha=0.7)

    plt.title("Kullanıcı Segmentasyonu (PCA + KMeans)")
    plt.xlabel("Ana Bileşen 1")
    plt.ylabel("Ana Bileşen 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def loo_segment_knn(user_summary, user_item_sparse, user_encoder, product_encoder, n_segments=4, k=10, test_limit=1000):
    # Segmentleme (KMeans)
    kmeans = KMeans(n_clusters=n_segments, random_state=42)
    user_segments = kmeans.fit_predict(user_item_sparse)

    # Her kullanıcı için segment bilgisi
    user_segment_df = pd.DataFrame({'UserID': user_encoder.classes_, 'segment': user_segments})

    hit_count = 0
    tested_count = 0
    results = []

    for user_id in tqdm(user_summary['UserID'][:test_limit]):
        if user_id not in user_encoder.classes_:
            continue

        target_idx = user_encoder.transform([user_id])[0]
        segment = user_segment_df[user_segment_df['UserID'] == user_id]['segment'].values[0]

        # Aynı segmentteki kullanıcıları seç
        segment_indices = user_segment_df[user_segment_df['segment'] == segment].index
        segment_matrix = user_item_sparse[segment_indices]

        actual_neighbors = min(k + 1, segment_matrix.shape[0])
        model = NearestNeighbors(n_neighbors=actual_neighbors)
        model.fit(segment_matrix)

        distances, indices = model.kneighbors([user_item_sparse[target_idx]])
        neighbors = segment_indices[indices.flatten()[1:]]

        recommended_items = user_item_sparse[neighbors].sum(axis=0).ravel()
        top_products = np.argsort(recommended_items)[::-1][:10]
        recommended_ids = product_encoder.inverse_transform(top_products)

        held_out = user_summary[user_summary['UserID'] == user_id]['ProductID'].values[0]
        hit = held_out in recommended_ids
        hit_count += int(hit)
        tested_count += 1

        results.append({'user_id': user_id, 'held_out': held_out, 'hit': hit, 'segment': segment})

    hit_rate = hit_count / tested_count if tested_count > 0 else 0
    return hit_rate, pd.DataFrame(results), user_segment_df

def loo_apriori_test(purchase_df, rules, test_limit=1000):
    hit_count = 0
    tested_count = 0
    results = []

    users = purchase_df['UserID'].unique()[:test_limit]

    for user_id in tqdm(users):
        user_items = purchase_df[purchase_df['UserID'] == user_id]['ProductID'].tolist()
        if len(user_items) < 2:
            continue

        held_out = user_items[-1]
        basket = set(user_items[:-1])
        
        recommended = set()
        for _, row in rules.iterrows():
            if set(row['antecedents']).issubset(basket):
                recommended.update(row['consequents'])

        hit = held_out in recommended
        hit_count += int(hit)
        tested_count += 1

        results.append({'user_id': user_id, 'held_out': held_out, 'hit': hit})

    hit_rate = hit_count / tested_count if tested_count > 0 else 0
    return hit_rate, pd.DataFrame(results)

def plot_hit_rate_comparison(knn_hit, apriori_hit, segment_hit, hybrid_hit):
    methods = ['KNN', 'Apriori', 'Segmented KNN', 'Hybrid']
    scores = [knn_hit, apriori_hit, segment_hit, hybrid_hit]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, [s * 100 for s in scores])
    plt.ylabel('Hit Rate (%)')
    plt.title('Öneri Sistemleri Başarı Karşılaştırması (LOO)')
    plt.ylim(0, max(scores) * 100 + 5)
    plt.grid(axis='y')
    plt.show()

def loo_knn_test(user_summary, user_item_sparse, user_encoder, product_encoder, k=10, test_limit=1000):
    hit_count = 0
    tested_count = 0
    results = []

    model = NearestNeighbors(n_neighbors=k+1)
    model.fit(user_item_sparse)

    for user_id in tqdm(user_summary['UserID'][:test_limit]):
        if user_id not in user_encoder.classes_:
            continue

        user_idx = user_encoder.transform([user_id])[0]

        distances, indices = model.kneighbors([user_item_sparse[user_idx]])
        neighbors = indices.flatten()[1:]
        neighbor_items = user_item_sparse[neighbors].sum(axis=0).ravel()
        top_products = np.argsort(neighbor_items)[::-1][:10]
        recommended_ids = product_encoder.inverse_transform(top_products)

        held_out = user_summary[user_summary['UserID'] == user_id]['ProductID'].values[0]
        hit = held_out in recommended_ids
        hit_count += int(hit)
        tested_count += 1

        results.append({'user_id': user_id, 'held_out': held_out, 'hit': hit})

    hit_rate = hit_count / tested_count if tested_count > 0 else 0
    return hit_rate, pd.DataFrame(results)
def loo_hybrid_test(user_summary, user_item_sparse, user_encoder, product_encoder,
                    rules, segment_results, user_segment_df, k=10, test_limit=1000):
    hit_count = 0
    tested_count = 0
    results = []

    model = NearestNeighbors(n_neighbors=k + 1)
    model.fit(user_item_sparse)

    for user_id in tqdm(user_summary['UserID'][:test_limit]):
        if user_id not in user_encoder.classes_:
            continue

        user_idx = user_encoder.transform([user_id])[0]
        held_out = user_summary[user_summary['UserID'] == user_id]['ProductID'].values[0]

        # --- KNN Önerileri ---
        distances, indices = model.kneighbors([user_item_sparse[user_idx]])
        neighbors = indices.flatten()[1:]
        knn_scores = user_item_sparse[neighbors].sum(axis=0)
        top_products = np.argsort(knn_scores)[::-1][:10]
        knn_items = product_encoder.inverse_transform(top_products).tolist()

        # --- Apriori Önerileri ---
        apriori_items = []
        for _, row in rules.iterrows():
            if set(row['antecedents']).issubset([held_out]):
                apriori_items.extend(list(row['consequents']))

        # --- Segment Önerileri (kendi segmentinden) ---
        segment = user_segment_df[user_segment_df['UserID'] == user_id]['segment'].values[0]
        segment_items = [prod for prod, _ in segment_results[segment]]

        # --- Hibrit Öneri ---
        hybrid = get_hybrid_recommendations(knn_items, apriori_items, segment_items)
        hybrid_ids = [item for item, _ in hybrid]

        hit = held_out in hybrid_ids
        hit_count += int(hit)
        tested_count += 1

        results.append({'user_id': user_id, 'held_out': held_out, 'hit': hit, 'segment': segment})

    hit_rate = hit_count / tested_count if tested_count > 0 else 0
    return hit_rate, pd.DataFrame(results)

def plot_segment_interactions(user_segment_df):
    # Gruplandırılmış ortalamaları hesapla
    avg_counts = user_segment_df.groupby('segment')[['product_view', 'add_to_cart', 'purchase']].mean()
    avg_counts.columns = ['view_count', 'cart_count', 'purchase_count']

    # Grafik
    ax = avg_counts.plot(kind='bar', figsize=(10, 6))
    plt.title("Segmentlere Göre Ortalama Etkileşimler")
    plt.xlabel("Segment")
    plt.ylabel("Ortalama Sayılar")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)