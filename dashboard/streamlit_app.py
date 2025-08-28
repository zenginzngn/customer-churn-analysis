#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Müşteri Terk Analizi Dashboard - Entegre Sürüm (Gün 17)
Model + Segmentasyon + Görseller
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional

# ------------------------------------------------------------------
# --- CONFIG ---
# ------------------------------------------------------------------
DATA_PATH = "segmented_df.csv"               # Segment sütunu da içeren veri
MODEL_PATH = "churn_model.pkl"               # Final model
SCALER_PATH = "scaler.pkl"                   # Opsiyonel
TRAIN_COLS_PATH = "training_columns.pkl"     # Opsiyonel; model eğitim sütun sırası
TARGET_COL = "Churn"
ID_COL = "customerID"
SEGMENT_COL = "Segment"

# ------------------------------------------------------------------
# --- STREAMLIT PAGE CONFIG ---
# ------------------------------------------------------------------
st.set_page_config(page_title="Müşteri Terk Analizi", layout="wide", page_icon="📊")

st.title("📊 Müşteri Terk Analizi – Entegre Dashboard")
st.caption("Gün 17: Model + Segmentasyon + Görselleştirme tek uygulamada.")


# ------------------------------------------------------------------
# --- CACHE'LENMİŞ YÜKLEYİCİLER ---
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_scaler(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_training_columns(path: str) -> Optional[List[str]]:
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ------------------------------------------------------------------
# --- DATA LOAD ---
# ------------------------------------------------------------------
df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)
training_cols = load_training_columns(TRAIN_COLS_PATH)

if df is None:
    st.error(f"Veri bulunamadı: `{DATA_PATH}`. Lütfen dosyayı dizine koyun.")
    st.stop()

# ------------------------------------------------------------------
# --- DATA SANITY / FALLBACKS ---
# ------------------------------------------------------------------
# AvgChargePerMonth yoksa hesapla
if "AvgChargePerMonth" not in df.columns and all(c in df.columns for c in ["TotalCharges", "tenure"]):
    tmp = df["tenure"].replace(0, np.nan)
    df["AvgChargePerMonth"] = df["TotalCharges"] / tmp
    df["AvgChargePerMonth"] = df["AvgChargePerMonth"].fillna(df["MonthlyCharges"])

# Her ihtimale karşı numeric kolon dönüşümü
for col in df.columns:
    if col not in [ID_COL, SEGMENT_COL] and df[col].dtype == "object":
        # binary objeler olabilir; coerce numeric
        maybe_num = pd.to_numeric(df[col], errors="coerce")
        # sayısalla dönüşen oran çoksa kabul et
        if maybe_num.notna().mean() > 0.8:
            df[col] = maybe_num

# NaN doldur (model güvenliği)
df = df.fillna(0)


# ------------------------------------------------------------------
# --- FEATURE COLS (model ile hizalama) ---
# ------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame, training_cols: Optional[List[str]] = None) -> List[str]:
    """Model eğitim kolonları varsa onları kullan; yoksa Churn/ID/Segment dışında kalan her şey."""
    if training_cols is not None:
        return training_cols
    
    drop_cols = [c for c in [TARGET_COL, ID_COL, SEGMENT_COL] if c in df.columns]
    return [c for c in df.columns if c not in drop_cols]

feature_cols = get_feature_columns(df, training_cols)

# Eğer model sklearn >=1.0 ve feature_names_in_ varsa, onu da kullanabiliriz
if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)

# Model ile dataframe kolon sayısı uyuşmuyor mu?
if model is not None:
    try:
        n_model_feats = getattr(model, "n_features_in_", None)
        if n_model_feats is not None and n_model_feats != len(feature_cols):
            st.warning(
                f"Model {n_model_feats} özellik bekliyor, ancak {len(feature_cols)} sütun seçildi. "
                "Eğitimde kullanılan sütun listesini kontrol edin."
            )
    except Exception:
        pass


# ------------------------------------------------------------------
# --- HELPER: Single-row feature vector ---
# ------------------------------------------------------------------
def prepare_features_for_model(row: pd.Series, feat_cols: List[str]) -> np.ndarray:
    vals = [row.get(c, 0) for c in feat_cols]
    return np.array(vals).reshape(1, -1)


# ------------------------------------------------------------------
# --- GLOBAL KPI (top row) ---
# ------------------------------------------------------------------
top1, top2, top3, top4 = st.columns(4)
with top1:
    st.metric("Toplam Müşteri", len(df))
with top2:
    if TARGET_COL in df.columns:
        st.metric("Genel Churn %", f"{df[TARGET_COL].mean():.1%}")
with top3:
    if "MonthlyCharges" in df.columns:
        st.metric("Ort. Aylık Ücret", f"${df['MonthlyCharges'].mean():.2f}")
with top4:
    if "tenure" in df.columns:
        st.metric("Ort. Tenure (Ay)", f"{df['tenure'].mean():.1f}")


# ------------------------------------------------------------------
# --- SIDEBAR MENU ---
# ------------------------------------------------------------------
menu = st.sidebar.radio(
    "Menü",
    ["🏠 Anasayfa", "🔍 Churn Tahmini", "📋 Segment Analizi", "📂 Tüm Müşteriler & Dışa Aktarım", "ℹ️ Hakkında", "➕ Yeni Müşteri Ekle"]
)


# ------------------------------------------------------------------
# --- PAGE: Anasayfa ---
# ------------------------------------------------------------------
if menu == "🏠 Anasayfa":
    st.subheader("Proje Özeti")
    st.write(
        """
        Bu dashboard, müşteri terk (churn) tahmini, segment bazlı analiz ve iş içgörüleri üretmek için geliştirilmiştir.
        
        **Bölümler:**
        - **Churn Tahmini:** Müşteri ID seç, terk olasılığını gör.
        - **Segment Analizi:** Segment profilleri, churn oranları, ücret dağılımları.
        - **Tüm Müşteriler:** Toplu tablo, risk sıralaması, dışa aktarım.
        """
    )
    if model is not None and hasattr(model, "feature_importances_"):
        st.write("Model: RandomForest benzeri; özellik önem dereceleri görüntülenebilir.")


# ------------------------------------------------------------------
# --- PAGE: Churn Tahmini ---
# ------------------------------------------------------------------
elif menu == "🔍 Churn Tahmini":
    st.subheader("🔍 Müşteri Bazlı Churn Tahmini")

    if ID_COL not in df.columns:
        st.error(f"'{ID_COL}' kolonu veri setinde yok, müşteri seçilemez.")
    else:
        customer_id = st.selectbox("Müşteri Seç:", options=df[ID_COL].unique())
        if st.button("Tahmin Et"):
            row = df[df[ID_COL] == customer_id].iloc[0]
            X_input = prepare_features_for_model(row, feature_cols)

            # scaler varsa uygula
            if scaler is not None:
                X_input = scaler.transform(X_input)

            if model is not None and hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_input)[0, 1]
                st.metric("Churn Olasılığı", f"{prob:.1%}")

                # Bar Chart (prob vs 1-prob)
                chart_data = {
                    "Durum": ["Churn Olabilir", "Churn Olmaz"],
                    "Olasılık": [prob, 1 - prob]
                }
                fig_bar = px.bar(
                    chart_data,
                    x="Durum",
                    y="Olasılık",
                    color="Durum",
                    text="Olasılık",
                    title="Tahmin Olasılığı"
                )
                fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig_bar.update_yaxes(range=[0,1])
                st.plotly_chart(fig_bar, use_container_width=True)

            elif model is not None:
                pred = model.predict(X_input)[0]
                st.metric("Churn Tahmini", int(pred))
            else:
                st.warning("Model yüklü değil; tahmin yapılamıyor.")

            # Gerçek etiket
            if TARGET_COL in df.columns:
                actual = row[TARGET_COL]
                st.write(f"**Gerçek Churn:** {actual}")

            # Özellikler tablosu
            with st.expander("Müşteri Özelliklerini Göster"):
                st.dataframe(row[feature_cols].to_frame(name="Değer"))

    # Genel churn oranı (Pie / Donut)
    if TARGET_COL in df.columns:
        churn_counts = df[TARGET_COL].value_counts()
        labels = ["Churn Olmaz", "Churn Olur"] if len(churn_counts) == 2 else churn_counts.index.astype(str)
        fig_pie = px.pie(
            names=labels,
            values=churn_counts.values,
            title="Genel Churn Oranı",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ------------------------------------------------------------------
# --- PAGE: Segment Analizi ---
# ------------------------------------------------------------------
elif menu == "📋 Segment Analizi":
    st.subheader("📋 Segment Bazlı Analiz")

    if SEGMENT_COL not in df.columns:
        st.error(f"'{SEGMENT_COL}' kolonu veri setinde bulunamadı.")
    else:
        segs = sorted(df[SEGMENT_COL].unique())
        seg_choice = st.selectbox("Segment Seçiniz:", segs)
        seg_df = df[df[SEGMENT_COL] == seg_choice].copy()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Müşteri Sayısı", len(seg_df))
        with c2:
            st.metric("Ort. Tenure", f"{seg_df['tenure'].mean():.1f}")
        with c3:
            st.metric("Ort. Ücret", f"${seg_df['MonthlyCharges'].mean():.2f}")
        with c4:
            if TARGET_COL in seg_df.columns:
                st.metric("Churn %", f"{seg_df[TARGET_COL].mean():.1%}")

        st.write("---")

        # Segment Summary Full
        if TARGET_COL in df.columns:
            segment_summary = (
                df.groupby(SEGMENT_COL)
                  .agg(
                      CustomerCount=(ID_COL, "count"),
                      AvgMonthlyCharges=("MonthlyCharges", "mean"),
                      ChurnRate=(TARGET_COL, "mean"),
                      AvgTenure=("tenure", "mean")
                  )
                  .reset_index()
            )
            st.write("### Segment Özeti")
            st.dataframe(segment_summary.style.format({
                "AvgMonthlyCharges": "{:.2f}",
                "ChurnRate": "{:.1%}",
                "AvgTenure": "{:.1f}"
            }))

            # Müşteri sayısı
            fig_cnt = px.bar(
                segment_summary,
                x="Segment",
                y="CustomerCount",
                title="Segment Bazlı Müşteri Sayısı",
                color="Segment"
            )
            st.plotly_chart(fig_cnt, use_container_width=True)

            # Ortalama ücret
            fig_charge = px.bar(
                segment_summary,
                x="Segment",
                y="AvgMonthlyCharges",
                title="Segment Bazlı Ortalama Aylık Ücret",
                color="Segment",
                labels={"AvgMonthlyCharges": "Ortalama Ücret"}
            )
            st.plotly_chart(fig_charge, use_container_width=True)

            # Churn Rate
            fig_churn = px.bar(
                segment_summary,
                x="Segment",
                y="ChurnRate",
                title="Segmentlere Göre Churn Oranı",
                color="Segment",
                labels={"ChurnRate": "Churn Oranı"}
            )
            fig_churn.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_churn, use_container_width=True)

        # Seçilen segment detayları
        with st.expander(f"Segment {seg_choice} – Detaylı Müşteri Listesi"):
            st.dataframe(seg_df)


# ------------------------------------------------------------------
# --- PAGE: Tüm Müşteriler & Dışa Aktarım ---
# ------------------------------------------------------------------
elif menu == "📂 Tüm Müşteriler & Dışa Aktarım":
    st.subheader("📂 Tüm Müşteriler & Toplu Risk Skoru")

    # Tüm müşteriler için tahmin (model varsa)
    if model is not None and hasattr(model, "predict_proba"):
        # Model girdisini hazırlamak için DataFrame subset
        X_all = df[feature_cols].copy()
        if scaler is not None:
            X_all = scaler.transform(X_all)
        probs = model.predict_proba(X_all)[:, 1]
        df["_ChurnProb"] = probs
        df["_ChurnRank"] = df["_ChurnProb"].rank(ascending=False, method="dense").astype(int)

        st.write("En Riskli İlk 20 Müşteri:")
        top_risk = df.sort_values("_ChurnProb", ascending=False).head(20)
        st.dataframe(
            top_risk[[ID_COL, "_ChurnProb", SEGMENT_COL] + ([TARGET_COL] if TARGET_COL in df.columns else [])]
            .rename(columns={"_ChurnProb": "ChurnProb"})
            .style.format({"ChurnProb": "{:.1%}"})
        )

        # İndirilebilir CSV
        csv_bytes = top_risk.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Riskli Müşterileri CSV Olarak İndir",
            data=csv_bytes,
            file_name="top_risk_customers.csv",
            mime="text/csv"
        )
    else:
        st.warning("Model yüklü değil veya olasılık tahmini desteklemiyor. Listeleme yapılamadı.")

    with st.expander("Tüm Veri Setini Göster"):
        st.dataframe(df.head(200))


# ------------------------------------------------------------------
# --- PAGE: About ---
# ------------------------------------------------------------------
elif menu == "ℹ️ Hakkında":
    st.subheader("Hakkında / Teknik Notlar")
    st.write(
        """
        **Proje:** Müşteri Terk (Churn) Analizi + Segmentasyon + Dashboard  
        **Veri:** Telco müşteri veri seti + türetilmiş değişkenler  
        **Model:** scikit-learn tabanlı (RandomForest / seçili final model)  
        **Geliştirici:** [Adını buraya ekle]  
        
        Bu uygulama staj sürecinde geliştirilmiş olup eğitim amaçlıdır.
        """
    )

    # Feature Importances (varsa)
    if model is not None and hasattr(model, "feature_importances_"):
        st.write("### Model Feature Importance")
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        fig_fi = px.bar(
            fi.head(20),
            x=fi.head(20).values,
            y=fi.head(20).index,
            orientation="h",
            title="En Önemli 20 Özellik",
            labels={"x": "Önem Skoru", "y": "Özellik"}
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Model feature_importances_ sağlamıyor veya desteklemiyor.")

# ------------------------------------------------------------------
# --- PAGE: Yeni Müşteri Ekle ---
# ------------------------------------------------------------------
elif menu == "➕ Yeni Müşteri Ekle":
    st.header("🆕 Yeni Müşteri Ekle")
    st.markdown("Lütfen yeni müşteri bilgilerini aşağıdaki form aracılığıyla giriniz:")

    with st.form("add_customer_form"):
        # Zorunlu Alanlar
        customer_id = st.text_input("Müşteri ID", placeholder="Örnek: new_customer_1")
        gender = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
        age = st.number_input("Yaş", min_value=18, max_value=100, value=30)
        tenure = st.number_input("Müşteri Süresi (yıl)", min_value=0, max_value=10, value=1)
        balance = st.number_input("Bakiye", min_value=0.0, value=5000.0)
        products_number = st.selectbox("Ürün Sayısı", [1, 2, 3, 4])
        has_cr_card = st.selectbox("Kredi Kartı Var mı?", ["Evet", "Hayır"])
        is_active_member = st.selectbox("Aktif Müşteri mi?", ["Evet", "Hayır"])
        estimated_salary = st.number_input("Tahmini Maaş", min_value=0.0, value=3000.0)

        # Opsiyonel Alanlar
        st.markdown("**Opsiyonel Bilgiler:**")
        senior_citizen = st.selectbox("Yaşlı Vatandaş mı? (SeniorCitizen)", ["", "Evet", "Hayır"])
        partner = st.selectbox("Evli mi? (Partner)", ["", "Evet", "Hayır"])
        dependents = st.selectbox("Bakmakla Yükümlü Olduğu Kişi Var mı? (Dependents)", ["", "Evet", "Hayır"])
        phone_service = st.selectbox("Telefon Servisi Var mı?", ["", "Evet", "Hayır"])
        online_security = st.selectbox("Online Güvenlik Hizmeti", ["", "Evet", "Hayır"])
        online_backup = st.selectbox("Online Yedekleme", ["", "Evet", "Hayır"])
        device_protection = st.selectbox("Cihaz Koruması", ["", "Evet", "Hayır"])
        tech_support = st.selectbox("Teknik Destek", ["", "Evet", "Hayır"])
        streaming_tv = st.selectbox("TV Streaming", ["", "Evet", "Hayır"])
        streaming_movies = st.selectbox("Film Streaming", ["", "Evet", "Hayır"])
        paperless_billing = st.selectbox("Kağıtsız Faturalama", ["", "Evet", "Hayır"])
        monthly_charges = st.number_input("Aylık Ücret (MonthlyCharges)", value=0.0)
        total_charges = st.number_input("Toplam Ücret (TotalCharges)", value=0.0)
        avg_charge_per_month = st.number_input("Aylık Ortalama Ücret", value=0.0)
        churn = st.selectbox("Churn (Terk Etti mi?)", ["", "Evet", "Hayır"])
        segment = st.selectbox("Segment", ["", "0", "1", "2"])
        contract = st.selectbox("Kontrat Tipi", ["", "Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("İnternet Servisi", ["", "DSL", "Fiber optic", "No"])
        multiple_lines = st.selectbox("Birden Fazla Hat", ["", "No", "No phone service", "Yes"])
        payment_method = st.selectbox("Ödeme Yöntemi", ["", "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])

        submitted = st.form_submit_button("Kaydet")

        if submitted:
            if customer_id.strip() == "":
                st.error("⚠️ Müşteri ID boş olamaz.")
            else:
                # Temel alanlar (zorunlu)
                new_customer = {
                    "CustomerId": customer_id,
                    "Gender": gender,
                    "Age": age,
                    "Tenure": tenure,
                    "Balance": balance,
                    "NumOfProducts": products_number,
                    "HasCrCard": 1 if has_cr_card == "Evet" else 0,
                    "IsActiveMember": 1 if is_active_member == "Evet" else 0,
                    "EstimatedSalary": estimated_salary
                }

                # Opsiyonel alanlar (varsa ekle)
                optional_fields = {
                    "SeniorCitizen": 1 if senior_citizen == "Evet" else (0 if senior_citizen == "Hayır" else ""),
                    "Partner": 1.0 if partner == "Evet" else (0.0 if partner == "Hayır" else ""),
                    "Dependents": 1.0 if dependents == "Evet" else (0.0 if dependents == "Hayır" else ""),
                    "PhoneService": 1.0 if phone_service == "Evet" else (0.0 if phone_service == "Hayır" else ""),
                    "OnlineSecurity": 1.0 if online_security == "Evet" else (0.0 if online_security == "Hayır" else ""),
                    "OnlineBackup": 1.0 if online_backup == "Evet" else (0.0 if online_backup == "Hayır" else ""),
                    "DeviceProtection": 1.0 if device_protection == "Evet" else (0.0 if device_protection == "Hayır" else ""),
                    "TechSupport": 1.0 if tech_support == "Evet" else (0.0 if tech_support == "Hayır" else ""),
                    "StreamingTV": 1.0 if streaming_tv == "Evet" else (0.0 if streaming_tv == "Hayır" else ""),
                    "StreamingMovies": 1.0 if streaming_movies == "Evet" else (0.0 if streaming_movies == "Hayır" else ""),
                    "PaperlessBilling": 1.0 if paperless_billing == "Evet" else (0.0 if paperless_billing == "Hayır" else ""),
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                    "AvgChargePerMonth": avg_charge_per_month,
                    "Churn": True if churn == "Evet" else (False if churn == "Hayır" else ""),
                    "Segment": segment if segment != "" else "",
                    "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
                    "Contract_One year": 1 if contract == "One year" else 0,
                    "Contract_Two year": 1 if contract == "Two year" else 0,
                    "InternetService_DSL": 1 if internet_service == "DSL" else 0,
                    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
                    "InternetService_No": 1 if internet_service == "No" else 0,
                    "MultipleLines_No": 1 if multiple_lines == "No" else 0,
                    "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
                    "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
                    "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer (automatic)" else 0,
                    "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
                    "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
                    "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0
                }

                new_customer.update(optional_fields)

                # Veriyi oku ve ekle
                df = pd.read_csv("segmented_df.csv")
                df = pd.concat([df, pd.DataFrame([new_customer])], ignore_index=True)
                df.to_csv("segmented_df.csv", index=False)

                st.success(f"✅ {customer_id} başarıyla eklendi.")
                st.rerun()  # Sayfayı yenileyerek veri setini tekrar yükle
