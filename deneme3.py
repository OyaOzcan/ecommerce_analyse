import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from deneme2 import *

st.set_page_config(page_title="📦 E-Ticaret Öneri ve Tahmin Sistemi", layout="wide")
st.title("📦 E-Ticaret Öneri ve Satın Alma Tahmin Sistemi")

# Sol Panel Ayarları
with st.sidebar:
    st.header("🔍 Model ve Öneri Seçimi")
    task_type = st.radio("Yapılacak işlem:", ["Satın Alma Tahmini", "Ürün Öneri Sistemi"])
    model_choice = st.selectbox("Model Seçimi", ["Logistic Regression", "Random Forest"])
    test_size = st.slider("Test Oranı", 0.1, 0.5, 0.3, 0.01)
    run_button = st.button("🚀 Modeli Eğit ve Değerlendir")

# Veri Yükleme
uploaded_file = st.file_uploader("📂 CSV veri dosyasını yükleyin", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Veri başarıyla yüklendi.")
    user_product_stats, user_encoder, product_encoder = create_user_product_features(df)

    if task_type == "Satın Alma Tahmini" and run_button:
        X_train, X_test, y_train, y_test = train_test_split(
            user_product_stats[["view_count", "cart_count"]],
            user_product_stats["Label"],
            test_size=test_size,
            random_state=42
        )
        with st.spinner("Model eğitiliyor ve tahmin yapılıyor..."):
            start = time.time()
            results = train_classification_models(X_train, X_test, y_train, y_test)
            duration = time.time() - start

        st.subheader("📊 Başarı Metrikleri")
        metrics = results[model_choice]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
        col2.metric("Precision", f"{metrics['Precision']:.2f}")
        col3.metric("Recall", f"{metrics['Recall']:.2f}")
        col4.metric("F1-Score", f"{metrics['F1-score']:.2f}")

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
        st.info(f"⏱️ Çalışma süresi: {duration:.2f} saniye")

    elif task_type == "Ürün Öneri Sistemi":
        tabs = st.tabs(["KNN", "Apriori", "Segment", "Hibrit", "Karşılaştırma"])

        with tabs[0]:
            st.subheader("👥 KNN Tabanlı Öneri")
            user_id = st.number_input("Kullanıcı ID (encoded)", min_value=0, step=1)
            if st.button("📌 Önerileri Getir"):
                try:
                    user_encoder = LabelEncoder().fit(df['UserID'])
                    product_encoder = LabelEncoder().fit(df['ProductID'])
                    df['UserID_enc'] = user_encoder.transform(df['UserID'])
                    df['ProductID_enc'] = product_encoder.transform(df['ProductID'])
                    purchases = df[df['EventType'] == 'purchase']
                    rows = purchases['UserID_enc']
                    cols = purchases['ProductID_enc']
                    data = [1] * len(purchases)
                    user_item_sparse = csr_matrix((data, (rows, cols)))
                    recs = oner_knn_ile_gelistirilmis(user_id, df, user_item_sparse, user_encoder, product_encoder)
                    if recs is not None:
                        st.write(recs)
                except Exception as e:
                    st.error(f"Hata: {str(e)}")

        with tabs[1]:
            st.subheader("🔗 Apriori Kuralları")
            user_encoder = LabelEncoder().fit(df['UserID'])
            product_encoder = LabelEncoder().fit(df['ProductID'])
            df['UserID_enc'] = user_encoder.transform(df['UserID'])
            df['ProductID_enc'] = product_encoder.transform(df['ProductID'])
            if st.button("Apriori Analizini Başlat"):
                itemsets, rules = run_apriori_analysis(df)
                if rules is not None:
                    st.dataframe(rules.sort_values("lift", ascending=False).head(10))
                else:
                    st.warning("Kurallar üretilemedi.")

        with tabs[2]:
            st.subheader("👥 Segment Bazlı Öneri")
            user_encoder = LabelEncoder().fit(df['UserID'])
            product_encoder = LabelEncoder().fit(df['ProductID'])
            df['UserID_enc'] = user_encoder.transform(df['UserID'])
            df['ProductID_enc'] = product_encoder.transform(df['ProductID'])
            summary, _, _ = segment_users(user_product_stats)

            user_id_seg = st.number_input("Kullanıcı ID (segment için, encoded)", min_value=0, step=1)
            if st.button("📌 Segment Önerilerini Getir"):
                purchases = df[df['EventType'] == 'purchase']
                rows = purchases['UserID_enc']
                cols = purchases['ProductID_enc']
                data = [1] * len(purchases)
                user_item_sparse = csr_matrix((data, (rows, cols)))
                recs = oner_knn_segmentli(user_id_seg, summary, df, user_item_sparse, user_encoder, product_encoder)
                if recs is not None:
                    st.write(recs)

            if st.checkbox("📊 Segment Etkileşim Grafiği ve Popüler Ürünler"):
                st.write("### Kullanıcı Etkileşimleri Segmentlere Göre")
                segment_summary = user_product_stats.merge(summary[['UserID', 'segment']], on='UserID')
                grouped = segment_summary.groupby('segment')[['view_count', 'cart_count', 'purchase_count']].mean()
                fig, ax = plt.subplots()
                grouped.plot(kind='bar', ax=ax)
                plt.title("Segmentlere Göre Ortalama Etkileşimler")
                st.pyplot(fig)

                st.write("### Segmentlere Göre Popüler Ürünler")
                df_seg = df.merge(summary[['UserID', 'segment']], on='UserID')
                pop_tab = st.tabs(["Segment 0", "Segment 1", "Segment 2"])
                for i in range(3):
                    with pop_tab[i]:
                        seg_df = df_seg[(df_seg['segment'] == i) & (df_seg['EventType'] == 'purchase')]
                        top_products = seg_df['ProductID'].value_counts().head(10)
                        st.write(top_products)


        with tabs[3]:
            st.subheader("🧪 Hibrit Öneri Sistemi")
            user_encoder = LabelEncoder().fit(df['UserID'])
            product_encoder = LabelEncoder().fit(df['ProductID'])
            df['UserID_enc'] = user_encoder.transform(df['UserID'])
            df['ProductID_enc'] = product_encoder.transform(df['ProductID'])
            purchases = df[df['EventType'] == 'purchase']
            rows = purchases['UserID_enc']
            cols = purchases['ProductID_enc']
            data = [1] * len(purchases)
            user_item_sparse = csr_matrix((data, (rows, cols)))

            user_id_hybrid = st.number_input("Kullanıcı ID (Hibrit Öneri)", min_value=0, step=1)
            if st.button("🚀 Hibrit Önerileri Getir"):
                try:
                    rec_knn = oner_knn_ile_gelistirilmis(user_id_hybrid, df, user_item_sparse, user_encoder, product_encoder)
                    summary, _, _ = segment_users(user_product_stats)
                    rec_seg = oner_knn_segmentli(user_id_hybrid, summary, df, user_item_sparse, user_encoder, product_encoder)
                    hybrid_products = pd.concat([rec_knn, rec_seg]).drop_duplicates().reset_index(drop=True)
                    st.write("### Hibrit Önerilen Ürünler")
                    st.write(hybrid_products)
                except Exception as e:
                    st.error(f"Hibrit öneri hatası: {str(e)}")


        with tabs[4]:
            st.subheader("📊 Başarı Karşılaştırması")
            st.write("LOO Hit Rate ölçümleri karşılaştırılıyor...")

            user_encoder = LabelEncoder().fit(df['UserID'])
            product_encoder = LabelEncoder().fit(df['ProductID'])
            df['UserID_enc'] = user_encoder.transform(df['UserID'])
            df['ProductID_enc'] = product_encoder.transform(df['ProductID'])
            purchases = df[df['EventType'] == 'purchase']
            rows = purchases['UserID_enc']
            cols = purchases['ProductID_enc']
            data = [1] * len(purchases)
            user_item_sparse = csr_matrix((data, (rows, cols)))

            with st.spinner("LOO testleri çalıştırılıyor..."):
                knn_result = test_knn_loo_success(purchases, user_item_sparse, user_encoder, product_encoder, test_limit=300)
                knn_hit = knn_result['hit'].mean()

                summary, _, _ = segment_users(user_product_stats)
                seg_result = test_knn_segmentli_loo(user_product_stats, summary, user_item_sparse, user_encoder, product_encoder, test_limit=300)
                seg_hit = seg_result['hit'].mean()

            comp_df = pd.DataFrame({
                'Algoritma': ['KNN', 'Segment'],
                'Hit Rate': [knn_hit, seg_hit]
            })
            st.dataframe(comp_df.set_index('Algoritma'))

            fig, ax = plt.subplots()
            sns.barplot(data=comp_df, x='Algoritma', y='Hit Rate', palette='viridis', ax=ax)
            plt.title("Algoritma Bazlı Hit Rate Karşılaştırması")
            st.pyplot(fig)
        