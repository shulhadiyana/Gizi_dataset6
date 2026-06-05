import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kalori Makanan",
    page_icon="🍽️",
    layout="wide"
)

# Title
st.title("🍽️ Prediksi Kalori Makanan")
st.markdown("Estimasi kalori makanan berdasarkan kandungan makronutrien dan mikronutrien")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('nilai_gizi_cleaned.csv')
    return df

# Train model
@st.cache_resource
def train_model():
    df = load_data()
    
    # Feature engineering
    df['protein_fat_interaction'] = df['protein_g'] * df['fat_g']
    df['carb_sugar_interaction'] = df['carbohydrate_g'] * df['sugar_g']
    
    # Fitur dan target
    feature_cols = ['protein_g', 'carbohydrate_g', 'fat_g', 'sugar_g',
                    'sodium_mg', 'fiber_g', 'usia',
                    'protein_fat_interaction', 'carb_sugar_interaction']
    
    X = df[feature_cols]
    y = df['kalori']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, scaler, feature_cols, r2, mae

# Load data
try:
    df = load_data()
    st.success(f"✅ Dataset loaded: {len(df)} makanan")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Train model
with st.spinner("Melatih model prediksi..."):
    try:
        model, scaler, feature_cols, r2, mae = train_model()
        st.success(f"✅ Model siap! R² Score: {r2:.4f}, MAE: {mae:.2f} kkal")
    except Exception as e:
        st.error(f"❌ Error training model: {e}")
        st.stop()

# Sidebar untuk input
st.sidebar.header("📊 Input Kandungan Gizi")

protein = st.sidebar.number_input("Protein (gram)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
carbohydrate = st.sidebar.number_input("Karbohidrat (gram)", min_value=0.0, max_value=300.0, value=20.0, step=1.0)
fat = st.sidebar.number_input("Lemak (gram)", min_value=0.0, max_value=150.0, value=5.0, step=1.0)
sugar = st.sidebar.number_input("Gula (gram)", min_value=0.0, max_value=150.0, value=2.0, step=1.0)
fiber = st.sidebar.number_input("Serat (gram)", min_value=0.0, max_value=50.0, value=3.0, step=1.0)
sodium = st.sidebar.number_input("Sodium (mg)", min_value=0.0, max_value=5000.0, value=100.0, step=50.0)
usia = st.sidebar.slider("Usia (tahun)", min_value=10, max_value=80, value=30)

# Tombol prediksi
predict_button = st.sidebar.button("🔮 Prediksi Kalori", use_container_width=True)

# Tampilkan ringkasan input
st.subheader("📋 Ringkasan Kandungan Gizi")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🥩 Protein", f"{protein} g", f"{protein*4} kkal")
with col2:
    st.metric("🍚 Karbohidrat", f"{carbohydrate} g", f"{carbohydrate*4} kkal")
with col3:
    st.metric("🧈 Lemak", f"{fat} g", f"{fat*9} kkal")
with col4:
    st.metric("👤 Usia", f"{usia} tahun", None)

# Preview data
with st.expander("📊 Preview Dataset"):
    st.dataframe(df.head(10))
    st.caption(f"Total data: {len(df)} makanan")

# Prediksi
if predict_button:
    with st.spinner("Menghitung prediksi kalori..."):
        # Buat dataframe untuk prediksi
        input_data = pd.DataFrame([[
            protein, carbohydrate, fat, sugar, sodium, fiber, usia,
            protein * fat,  # protein_fat_interaction
            carbohydrate * sugar  # carb_sugar_interaction
        ]], columns=feature_cols)
        
        # Scaling dan prediksi
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # Kalori dari rumus dasar
        kalori_manual = (protein * 4) + (carbohydrate * 4) + (fat * 9)
        
        # Tampilkan hasil
        st.markdown("---")
        st.subheader("✨ Hasil Prediksi")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 10px; text-align: center; color: white;">
                <h2>Total Kalori</h2>
                <h1 style="font-size: 4rem;">{prediction:.0f}</h1>
                <h3>kkal per sajian</h3>
                <hr>
                <p>📊 Kalori berdasarkan rumus standar (4-4-9): <b>{kalori_manual:.0f} kkal</b></p>
                <p>🤖 Kalori prediksi Machine Learning: <b>{prediction:.0f} kkal</b></p>
                <p>🎯 Akurasi Model: <b>R² = {r2:.3f}</b> | MAE = {mae:.1f} kkal</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Rekomendasi
        st.subheader("💡 Rekomendasi")
        if prediction < 100:
            st.success("✅ **Makanan rendah kalori** - Cocok untuk camilan sehat atau diet.")
        elif prediction < 300:
            st.success("✅ **Makanan kalori sedang** - Pilihan baik untuk makanan utama.")
        elif prediction < 500:
            st.warning("⚠️ **Makanan kalori tinggi** - Perhatikan porsi konsumsi.")
        else:
            st.error("🔥 **Makanan sangat tinggi kalori** - Konsumsi dalam porsi kecil.")
        
        # Grafik perbandingan
        st.subheader("📊 Perbandingan Komposisi")
        
        # Data untuk chart
        macro_data = pd.DataFrame({
            'Nutrisi': ['Protein', 'Karbohidrat', 'Lemak'],
            'Gram': [protein, carbohydrate, fat],
            'Kalori': [protein * 4, carbohydrate * 4, fat * 9]
        })
        
        fig1 = px.bar(macro_data, x='Nutrisi', y='Gram', 
                      color='Nutrisi', title="Komposisi Makronutrien (gram)",
                      color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'])
        fig1.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig1, use_container_width=True)

# Informasi model
with st.sidebar.expander("📖 Tentang Model"):
    st.markdown(f"""
    **Model**: Random Forest Regressor
    
    **Performa Model**:
    - R² Score: {r2:.4f}
    - MAE: {mae:.2f} kkal
    
    **Fitur yang digunakan**:
    - Protein, Karbohidrat, Lemak
    - Gula, Sodium, Serat
    - Usia
    - Interaksi Protein×Lemak
    - Interaksi Karbohidrat×Gula
    
    **Data Latih**: {len(df)} sampel makanan
    """)

# Footer
st.markdown("---")
st.caption("🔬 Aplikasi ini menggunakan Random Forest Regression untuk memprediksi kalori makanan berdasarkan kandungan gizinya. Model dilatih dengan dataset nilai gizi makanan Indonesia.")
