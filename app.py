
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.set_page_config(page_title="Анализ Транзакций", layout="wide")

st.title("📊 Кластеризация банковских транзакций")
st.write("Это приложение использует модель KMeans для группировки транзакций по сумме и географии.")

# =========================
# Константы
# =========================
FEATURES = ['amount', 'long', 'lat']

# =========================
# Загрузка ресурсов
# =========================
@st.cache_resource
def load_model_and_data():
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv(r'c:\Users\admin\Desktop\project2\data\transactions.csv')
    
    # 🔥 ВАЖНО: то же переименование, что при обучении
    df = df.rename(columns={
        'transaction_dollar_amount': 'amount',
        'Long': 'long',
        'Lat': 'lat'
    })
    
    # Убираем пробелы и приводим к нижнему регистру (доп. защита)
    df.columns = df.columns.str.strip().str.lower()
    
    return model, scaler, df

try:
    model, scaler, df = load_model_and_data()
except FileNotFoundError:
    st.error("Ошибка: Файлы модели или данных не найдены. Сначала обучите модель.")
    st.stop()

# =========================
# Проверка колонок
# =========================
missing_cols = [col for col in FEATURES if col not in df.columns]
if missing_cols:
    st.error(f"Ошибка: отсутствуют колонки {missing_cols}")
    st.write("Доступные колонки:", df.columns.tolist())
    st.stop()

    # =========================
# Просмотр данных
# =========================
st.subheader("📄 Первые 10 строк датасета")

st.dataframe(df.head(10), use_container_width=True)

# =========================
# Sidebar ввод
# =========================
st.sidebar.header("Ввод данных транзакции")

def get_user_input():
    amount_str = st.sidebar.text_input("Сумма транзакции ($)", "100.0")
    long_str = st.sidebar.text_input("Долгота (Long)", "-80.0")
    lat_str = st.sidebar.text_input("Широта (Lat)", "40.0")
    
    try:
        data = np.array([[float(amount_str), float(long_str), float(lat_str)]])
        return data
    except ValueError:
        st.sidebar.error("Ошибка: введите число")
        return None

user_data = get_user_input()

if st.sidebar.button("Предсказать кластер"):
    if user_data is not None:
        scaled_data = scaler.transform(user_data)
        prediction = model.predict(scaled_data)
        st.sidebar.success(f"Результат: Кластер №{prediction[0]}")

# =========================
# Визуализация
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Географическое распределение")
    
    sample_df = df.sample(min(2000, len(df))).copy()
    sample_df = sample_df.dropna(subset=FEATURES)

    X_sample_scaled = scaler.transform(sample_df[FEATURES])
    sample_df['cluster'] = model.predict(X_sample_scaled)
    
    fig1, ax1 = plt.subplots()
    sns.scatterplot(
        data=sample_df,
        x='long',
        y='lat',
        hue='cluster',
        palette='viridis',
        ax=ax1
    )
    st.pyplot(fig1)

with col2:
    st.subheader("Визуализация PCA (2D)")
    
    df_clean = df.dropna(subset=FEATURES)
    X_all_scaled = scaler.transform(df_clean[FEATURES])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all_scaled)

    pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    pca_df['cluster'] = model.predict(X_all_scaled)
    
    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        data=pca_df.sample(min(2000, len(pca_df))),
        x='PCA1',
        y='PCA2',
        hue='cluster',
        palette='magma',
        ax=ax2
    )
    st.pyplot(fig2)

