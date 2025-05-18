import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from recommendation_utils import *

st.set_page_config(layout="wide")
st.title("📦 E-Ticaret Öneri ve Satın Alma Tahmin Sistemi")

uploaded_file = st.file_uploader("📂 CSV veri dosyasını yükleyin", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    stats, interaction_matrix, purchase_df, user_summary, user_encoder, product_encoder = prepare_datasets(df)

    # 🛑 NaN değerleri temizlemeden encoder kullanma!
    df = df[df["ProductID"].notnull() & df["UserID"].notnull()].copy()

    # ✅ Artık güvenle transform edebilirsin
    df["UserID_enc"] = user_encoder.transform(df["UserID"])
    df["ProductID_enc"] = product_encoder.transform(df["ProductID"])

    X = stats[["view_count", "cart_count"]]
    y = stats["Label"]


    st.sidebar.header("🔍 Model ve Öneri Seçimi")
    task = st.sidebar.radio("Yapılacak İşlem:", ["Satın Alma Tahmini", "Ürün Öneri Sistemi"], key="task_selector")

    if task == "Satın Alma Tahmini":
        model_choice = st.sidebar.selectbox("Model Seçimi", ["LogisticRegression", "RandomForest"])
        test_ratio = st.sidebar.slider("Test Oranı", 0.1, 0.5, 0.3)

        if st.sidebar.button("🚀 Modeli Eğit ve Değerlendir"):
            metrics, model = run_selected_model(X, y, model_name=model_choice, test_size=test_ratio)
            st.subheader("📊 Model Başarı Metrikleri")
            st.dataframe(metrics['report_df'])
            st.write("Confusion Matrix:")
            st.dataframe(metrics['confusion_matrix'])

    elif task == "Ürün Öneri Sistemi":
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["KNN", "Apriori", "Segment", "Hibrit", "Karşılaştırma"])

        with tab1:
            st.subheader("👥 KNN Tabanlı Öneri")
            selected_user = st.number_input("Kullanıcı ID (encoded)", min_value=0, max_value=interaction_matrix.shape[0]-1)

            if st.button("📌 Önerileri Getir"):
                knn_items = get_knn_recommendations(interaction_matrix, selected_user)
                df_knn = pd.DataFrame(knn_items, columns=["ProductID_enc", "Önerilme Sayısı"])
                df_knn["ProductID"] = product_encoder.inverse_transform(df_knn["ProductID_enc"])
                st.table(df_knn[["ProductID", "Önerilme Sayısı"]])

        with tab2:
            st.subheader("📦 Apriori Kuralları")
            rules_df = get_apriori_rules(df)
            if not rules_df.empty:
                st.dataframe(rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            else:
                st.warning("❗ Belirtilen eşiklerle kural üretilemedi.")

        with tab3:
            st.subheader("👤 Segment Bazlı Öneri")
            segment_results, user_segment_df = get_segment_recommendations(df)

            for seg_id, items in segment_results.items():
                st.markdown(f"### 🔹 Segment {seg_id} İçin En Popüler Ürünler")
                df_segment = pd.DataFrame(items, columns=["ProductID_enc", "Satın Alma Sayısı"])
                df_segment["ProductID"] = product_encoder.inverse_transform(df_segment["ProductID_enc"])
                st.dataframe(df_segment[["ProductID", "Satın Alma Sayısı"]])

            st.markdown("### 🎯 Segment Dağılımı (PCA ile 2D Görselleştirme)")
            plt.clf()
            plot_segments(interaction_matrix.values, n_clusters=len(segment_results))
            st.pyplot(plt)

            st.markdown("### 📊 Segmentlere Göre Etkileşim Ortalamaları")
            plt.clf()
            plot_segment_interactions(user_segment_df)

        with tab4:
            st.subheader("🔄 Hibrit Öneri")
            selected_user = st.number_input("Kullanıcı ID (encoded) - Hibrit", min_value=0, max_value=interaction_matrix.shape[0]-1, key='hybrid_user')

            knn_items = [item for item, _ in get_knn_recommendations(interaction_matrix, selected_user)]

            apriori_items = []
            basket = set(df[df["UserID_enc"] == selected_user]["ProductID_enc"])
            for _, row in rules_df.iterrows():
                if set(row['antecedents']).issubset(basket):
                    apriori_items.extend(list(row['consequents']))

            segment_items = [prod for seg, items in segment_results.items() for prod, _ in items]
            hybrid = get_hybrid_recommendations(knn_items, apriori_items, segment_items)
            hybrid_df = pd.DataFrame(hybrid, columns=["ProductID_enc", "Skor"])
            hybrid_df["ProductID"] = product_encoder.inverse_transform(hybrid_df["ProductID_enc"])
            st.table(hybrid_df[["ProductID", "Skor"]])

        with tab5:
            st.subheader("📈 LOO Hit Rate Karşılaştırması")
            user_item_sparse = interaction_matrix.values

            st.info("Bu işlem birkaç saniye sürebilir...")

           # Streamlit sidebar üzerinden kontrol edilebilir test sayısı (kullanıcıya seçtirebilirsin)
            test_limit = st.sidebar.slider("LOO Test Kullanıcı Sayısı", 100, 1000, 300, step=100)

            # LOO testleri
            knn_hit, _ = loo_knn_test(user_summary, user_item_sparse, user_encoder, product_encoder, test_limit=test_limit)
            apriori_hit, _ = loo_apriori_test(purchase_df, rules_df, test_limit=test_limit)
            segment_hit, _, user_segment_df = loo_segment_knn(
                user_summary, user_item_sparse, user_encoder, product_encoder,
                test_limit=test_limit
            )
            hybrid_hit, _ = loo_hybrid_test(
                user_summary, user_item_sparse, user_encoder, product_encoder,
                rules_df, segment_results, user_segment_df=user_segment_df,
                test_limit=test_limit
)

            st.success(f"🎯 KNN Hit Rate: {knn_hit:.2%}")
            st.success(f"🎯 Apriori Hit Rate: {apriori_hit:.2%}")
            st.success(f"🎯 Segment Hit Rate: {segment_hit:.2%}")
            st.success(f"🎯 Hybrid Hit Rate: {hybrid_hit:.2%}")

            plt.clf()
            plot_hit_rate_comparison(knn_hit, apriori_hit, segment_hit, hybrid_hit)
            st.pyplot(plt)

else:
    st.info("Lütfen bir CSV dosyası yükleyin.")
