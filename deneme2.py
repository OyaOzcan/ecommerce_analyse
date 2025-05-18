# 1. Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, coo_matrix
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import random

# 2. Veri yükleme
def load_data(path="ecommerce_clickstream_transactions.csv"):
    df = pd.read_csv(path)
    return df

# 3. Özellik çıkarımı
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

    # --- Encoder işlemleri ---
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    user_product_stats["UserID_enc"] = user_encoder.fit_transform(user_product_stats["UserID"])
    user_product_stats["ProductID_enc"] = product_encoder.fit_transform(user_product_stats["ProductID"])

    return user_product_stats, user_encoder, product_encoder

# 4. Segmentleme ve görselleştirme
def segment_users(user_product_stats):
    user_summary = user_product_stats.groupby("UserID")[["view_count", "cart_count", "purchase_count"]].sum().reset_index()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_summary[["view_count", "cart_count", "purchase_count"]])

    kmeans = KMeans(n_clusters=3, random_state=42)
    user_summary["segment"] = kmeans.fit_predict(X_scaled)

    user_summary.groupby("segment")[["view_count", "cart_count", "purchase_count"]].mean().plot(kind='bar')
    plt.title("Segmentlere Göre Ortalama Etkileşimler")
    plt.xlabel("Segment")
    plt.ylabel("Ortalama Sayılar")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return user_summary, scaler, kmeans

# 5. Temel KNN LOO testi
def test_knn_loo_success(purchases, user_item_sparse, user_encoder, product_encoder, test_limit=1000, k=5):
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

# 6. Geliştirilmiş KNN öneri

def oner_knn_ile_gelistirilmis(user_id, score_df, user_item_sparse, user_encoder, product_encoder, k=5):
    if user_id not in user_encoder.classes_:
        print(f"Kullanıcı {user_id} sistemde yok.")
        return

    target_user_index = user_encoder.transform([user_id])[0]

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_item_sparse)

    distances, indices = knn_model.kneighbors(user_item_sparse[target_user_index], n_neighbors=k+1)
    similar_user_indices = indices[0][1:]

    target_products = set(score_df[score_df['UserID_enc'] == target_user_index]['ProductID_enc'])

    recommended_products = set()
    for sim_user_index in similar_user_indices:
        sim_products = set(score_df[score_df['UserID_enc'] == sim_user_index]['ProductID_enc'])
        recommended_products.update(sim_products - target_products)

    if recommended_products:
        recommended_ids = product_encoder.inverse_transform(list(recommended_products))
        return pd.DataFrame(recommended_ids, columns=["KNN (Geliştirilmiş) Önerilen Ürünler"])
    else:
        print("Önerilecek yeni ürün bulunamadı.")
        return None

# 7. Geliştirilmiş KNN LOO testi
def test_knn_loo_gelistirilmis(score_df, user_item_sparse, user_encoder, product_encoder, test_limit=1000, k=10):
    hit_count = 0
    tested_count = 0
    results = []

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_item_sparse)

    for user_idx in range(min(test_limit, user_item_sparse.shape[0])):
        user_products = score_df[score_df['UserID_enc'] == user_idx]['ProductID_enc'].tolist()
        if len(user_products) < 2:
            continue

        held_out = random.choice(user_products)
        temp_vector = user_item_sparse[user_idx].toarray().copy()
        temp_vector[0, held_out] = 0

        distances, indices = knn_model.kneighbors(temp_vector, n_neighbors=k+1)
        similar_users = indices[0][1:]

        recommended_products = set()
        for sim_idx in similar_users:
            sim_products = score_df[score_df['UserID_enc'] == sim_idx]['ProductID_enc'].tolist()
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
    print(f"🎯 Geliştirilmiş KNN LOO Hit Rate: {hit_count}/{tested_count} = %{hit_rate * 100:.2f}")
    return pd.DataFrame(results)

# # 8. Segment bazlı öneri
# def oner_knn_segmentli(user_id, user_summary, score_df, user_item_sparse, user_encoder, product_encoder, k=10):
#     if user_id not in user_encoder.classes_:
#         print(f"Kullanıcı {user_id} sistemde yok.")
#         return

#     target_user_idx = user_encoder.transform([user_id])[0]

#     segment = user_summary[user_summary['UserID'] == user_id]['segment'].values[0]
#     same_segment_users = user_summary[user_summary['segment'] == segment]['UserID'].tolist()
#     same_segment_indices = user_encoder.transform([uid for uid in same_segment_users if uid in user_encoder.classes_])

#     segment_sparse = user_item_sparse[same_segment_indices]
#     knn = NearestNeighbors(metric='cosine', algorithm='brute')
#     knn.fit(segment_sparse)

#     relative_index = list(same_segment_indices).index(target_user_idx)
#     distances, indices = knn.kneighbors(segment_sparse[relative_index], n_neighbors=min(k+1, len(same_segment_indices)))

#     similar_relative_indices = indices[0][1:]
#     similar_user_indices = [same_segment_indices[i] for i in similar_relative_indices]

#     target_products = set(score_df[score_df['UserID_enc'] == target_user_idx]['ProductID_enc'])

#     recommended_products = set()
#     for sim_idx in similar_user_indices:
#         sim_products = set(score_df[score_df['UserID_enc'] == sim_idx]['ProductID_enc'])
#         recommended_products.update(sim_products - target_products)

#     if recommended_products:
#         recommended_ids = product_encoder.inverse_transform(list(recommended_products))
#         return pd.DataFrame(recommended_ids, columns=["Segment Bazlı KNN Önerilen Ürünler"])
#     else:
#         print("Önerilecek ürün bulunamadı.")
#         return None

def oner_knn_segmentli(user_id, user_summary, score_df, user_item_sparse, user_encoder, product_encoder, k=10):
    # Kullanıcı sistemde var mı kontrolü
    if user_id not in user_encoder.classes_:
        print(f"Kullanıcı {user_id} sistemde yok.")
        return

    # Encode edilmiş kullanıcı indexi
    target_user_idx = user_encoder.transform([user_id])[0]

    # Hedef kullanıcının segmentini al
    segment = user_summary[user_summary['UserID'] == user_id]['segment'].values[0]

    # Aynı segmentteki kullanıcıları bul
    same_segment_users = user_summary[user_summary['segment'] == segment]['UserID'].tolist()
    same_segment_indices = user_encoder.transform([uid for uid in same_segment_users if uid in user_encoder.classes_])

    # Aynı segmentin sparse matrisini oluştur
    segment_sparse = user_item_sparse[same_segment_indices]

    # KNN modeli sadece aynı segment için eğitiliyor
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(segment_sparse)

    # Hedef kullanıcının bu segment içindeki sırasını bul
    relative_index = list(same_segment_indices).index(target_user_idx)

    # Benzer kullanıcıları bul
    distances, indices = knn.kneighbors(segment_sparse[relative_index], n_neighbors=min(k+1, len(same_segment_indices)))

    # Kendisi hariç benzer kullanıcı indexleri (segment içindeki sıralama)
    similar_relative_indices = indices[0][1:]
    similar_user_indices = [same_segment_indices[i] for i in similar_relative_indices]

    # Hedef kullanıcının aldığı ürünleri al
    target_products = set(score_df[score_df['UserID_enc'] == target_user_idx]['ProductID_enc'])

    # Benzer kullanıcıların ürünlerinden öneri üret
    recommended_products = set()
    for sim_idx in similar_user_indices:
        sim_products = set(score_df[score_df['UserID_enc'] == sim_idx]['ProductID_enc'])
        recommended_products.update(sim_products - target_products)

    if recommended_products:
        recommended_ids = product_encoder.inverse_transform(list(recommended_products))
        return pd.DataFrame(recommended_ids, columns=["Segment Bazlı KNN Önerilen Ürünler"])
    else:
        print("Önerilecek ürün bulunamadı.")
        return

# 9. Apriori birliktelik analizi
def run_apriori_analysis(df, min_support=0.001, min_lift=1.0):
    purchase_df = df[df['EventType'] == 'purchase']
    transactions = purchase_df.groupby('UserID')['ProductID'].apply(list)

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    basket_encoded = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        print("❌ Sık öğe kümeleri bulunamadı. min_support değerini azaltmayı deneyin.")
        return None, None

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    return frequent_itemsets, rules

# 10. Cosine benzerlik ile kullanıcı ağı

def build_similarity_graph(df, threshold=0.5):
    purchase_df = df[df['EventType'] == 'purchase']

    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    purchase_df['UserID_enc'] = user_encoder.fit_transform(purchase_df['UserID'])
    purchase_df['ProductID_enc'] = product_encoder.fit_transform(purchase_df['ProductID'])

    rows = purchase_df['UserID_enc']
    cols = purchase_df['ProductID_enc']
    data = [1] * len(purchase_df)

    num_users = purchase_df['UserID_enc'].nunique()
    num_products = purchase_df['ProductID_enc'].nunique()

    user_item_sparse = coo_matrix((data, (rows, cols)), shape=(num_users, num_products))
    user_similarity_matrix = cosine_similarity(user_item_sparse)

    G = nx.Graph()
    for user_id in purchase_df['UserID_enc'].unique():
        G.add_node(user_id)

    for i in range(len(user_similarity_matrix)):
        for j in range(i+1, len(user_similarity_matrix)):
            if user_similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=user_similarity_matrix[i, j])

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    plt.title("Kullanıcılar Arası Etkileşim Ağı")
    plt.show()

    return G, user_encoder, product_encoder

def prepare_classification_data(user_product_stats):
    X = user_product_stats[["view_count", "cart_count"]]
    y = user_product_stats["Label"]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_classification_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Classification Report": classification_report(y_test, y_pred, output_dict=True)
        }
    return results

# def test_knn_segmentli_loo(user_product_stats, summary_df, user_item_sparse, user_encoder, product_encoder, test_limit=1000, k=10):
#     from sklearn.neighbors import NearestNeighbors
#     import random

#     hit_count = 0
#     tested_count = 0
#     results = []

#     for user_id in summary_df['UserID'][:test_limit]:
#         if user_id not in user_encoder.classes_:
#             continue

#         target_user_idx = user_encoder.transform([user_id])[0]
#         user_products = user_product_stats[user_product_stats['UserID'] == user_id]['ProductID'].tolist()
#         if len(user_products) < 2:
#             continue

#         held_out = product_encoder.transform([random.choice(user_products)])[0]
#         temp_vector = user_item_sparse[target_user_idx].toarray().copy()
#         temp_vector[0, held_out] = 0

#         segment = summary_df[summary_df['UserID'] == user_id]['segment'].values[0]
#         same_segment_users = summary_df[summary_df['segment'] == segment]['UserID'].tolist()
#         same_segment_indices = user_encoder.transform([uid for uid in same_segment_users if uid in user_encoder.classes_])

#         segment_sparse = user_item_sparse[same_segment_indices]
#         knn = NearestNeighbors(metric='cosine', algorithm='brute')
#         knn.fit(segment_sparse)

#         relative_index = list(same_segment_indices).index(target_user_idx)
#         distances, indices = knn.kneighbors(segment_sparse[relative_index], n_neighbors=min(k+1, len(same_segment_indices)))
#         similar_relative_indices = indices[0][1:]
#         similar_user_indices = [same_segment_indices[i] for i in similar_relative_indices]

#         recommended_products = set()
#         for sim_idx in similar_user_indices:
#             sim_products = user_product_stats[user_product_stats['UserID'] == user_encoder.inverse_transform([sim_idx])[0]]['ProductID'].tolist()
#             recommended_products.update(product_encoder.transform(sim_products))

#         hit = held_out in recommended_products
#         hit_count += int(hit)
#         tested_count += 1

#         results.append({
#             'user_id': user_id,
#             'held_out_product': product_encoder.inverse_transform([held_out])[0],
#             'hit': hit,
#             'recommended_count': len(recommended_products),
#             'segment': segment
#         })

#     return pd.DataFrame(results)

def test_knn_segmentli_loo(score_df, user_summary, user_item_sparse, user_encoder, product_encoder, test_limit=1000, k=10):
    from sklearn.neighbors import NearestNeighbors
    import random

    hit_count = 0
    tested_count = 0
    results = []

    for user_id in user_summary['UserID'][:test_limit]:
        if user_id not in user_encoder.classes_:
            continue

        target_user_idx = user_encoder.transform([user_id])[0]
        user_segment = user_summary[user_summary['UserID'] == user_id]['segment'].values[0]

        user_products = score_df[score_df['UserID_enc'] == target_user_idx]['ProductID_enc'].tolist()
        if len(user_products) < 2:
            continue

        held_out = random.choice(user_products)
        temp_vector = user_item_sparse[target_user_idx].toarray().copy()
        temp_vector[0, held_out] = 0

        # Segmentteki kullanıcılar
        same_segment_users = user_summary[user_summary['segment'] == user_segment]['UserID'].tolist()
        same_segment_indices = user_encoder.transform([uid for uid in same_segment_users if uid in user_encoder.classes_])

        if len(same_segment_indices) <= 1:
            continue

        segment_sparse = user_item_sparse[same_segment_indices]

        try:
            relative_index = list(same_segment_indices).index(target_user_idx)
        except ValueError:
            continue

        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(segment_sparse)

        distances, indices = knn.kneighbors(segment_sparse[relative_index], n_neighbors=min(k+1, len(same_segment_indices)))

        similar_relative_indices = indices[0][1:]
        similar_user_indices = [same_segment_indices[i] for i in similar_relative_indices]

        recommended_products = set()
        for sim_idx in similar_user_indices:
            sim_products = score_df[score_df['UserID_enc'] == sim_idx]['ProductID_enc'].tolist()
            recommended_products.update(sim_products)

        hit = held_out in recommended_products
        hit_count += int(hit)
        tested_count += 1

        results.append({
            'user_id': user_id,
            'held_out_product': product_encoder.inverse_transform([held_out])[0],
            'hit': hit,
            'recommended_count': len(recommended_products),
            'segment': user_segment
        })

    hit_rate = hit_count / tested_count if tested_count > 0 else 0
    print(f"🎯 Segment Bazlı KNN LOO Hit Rate: {hit_count}/{tested_count} = %{hit_rate * 100:.2f}")
    return pd.DataFrame(results)
