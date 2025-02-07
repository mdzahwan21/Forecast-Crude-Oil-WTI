import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from datetime import datetime, timedelta
import time  # Tambahkan untuk menunggu 1 detik
import plotly.express as px  # Untuk mengganti grafik menjadi Plotly
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------------------
# 1. Konfigurasi Halaman
# -------------------------------------
st.set_page_config(
    page_title="Prediksi Harga Minyak Mentah WTI",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------
# 2. Heading Aplikasi
# -------------------------------------
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0px;'>📈 Prediksi Harga Minyak Mentah WTI</h1>
    <p style='text-align: center; color: grey; margin-top: 0px;'>
    Menggunakan model LSTM berbasis data historis
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------------
# 3. Inisialisasi Halaman di session_state
# -------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"  # default halaman pertama

# -------------------------------------
# 4. Sidebar: Navigasi dengan Tombol
# -------------------------------------
st.sidebar.markdown("## Navigasi")
if st.sidebar.button("📊 Dashboard"):
    st.session_state["page"] = "Dashboard"
if st.sidebar.button("📝 Prediksi"):
    st.session_state["page"] = "Prediksi"

# -------------------------------------
# 5. Sidebar: Pengaturan & Tentang Aplikasi
# -------------------------------------
st.sidebar.title("⚙️ Pengaturan & Tentang Aplikasi")
with st.sidebar.expander("❓ Bagaimana cara kerja aplikasi ini?", expanded=True):
    st.write(
        """
        Aplikasi ini menggunakan **model LSTM** yang telah dilatih untuk memprediksi harga minyak mentah WTI 
        berdasarkan rangkaian waktu data historis.\n
        **Langkah-langkah** penggunaannya:
        1. Unggah file CSV berisi data historis minimal sebanyak 30 baris (Date, Price, Open, High, Low, Vol., Change %).
        2. Setelah file diunggah, klik tombol **Prediksi** yang akan muncul.
        3. Lihat hasil prediksi dan unduh jika diperlukan.
        """
    )

# ==================================================================
# FUNGSI-FUNGSI UTAMA (LOAD MODEL, SCALER, PREPROCESS, PREDICT)
# ==================================================================

@st.cache_resource
def load_trained_model(model_path="lstm_wti_model.h5"):
    """Memuat model LSTM yang telah dilatih."""
    return load_model(model_path, compile=False)

@st.cache_resource
def load_scaler(scaler_path="minmax_scaler.pkl"):
    """Memuat MinMaxScaler untuk normalisasi data."""
    return joblib.load(scaler_path)

def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan dan menormalisasi data input."""
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df.sort_values('Date', inplace=True)

    # Bersihkan "Change %"
    df["Change %"] = df["Change %"].str.replace('%','', regex=True).astype(float)

    # Convert Vol.
    def convert_volume(vol_str):
        if isinstance(vol_str, str):
            vol_str = vol_str.replace(',', '')
            if 'K' in vol_str:
                return float(vol_str.replace('K','')) * 1000
            elif 'M' in vol_str:
                return float(vol_str.replace('M','')) * 1000000
            return float(vol_str)
        return vol_str
    df['Vol.'] = df['Vol.'].apply(convert_volume)

    # 🔹 Pengecekan sebelum interpolasi
    st.write("📊 Jumlah data sebelum interpolasi:", len(df))

    # Interpolasi data agar berurutan harian
    df.set_index('Date', inplace=True)
    df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='D'))
    df.interpolate(method='linear', inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)

    # 🔹 Pengecekan setelah interpolasi
    st.write("📊 Jumlah data setelah interpolasi:", len(df))

    # Kita pakai placeholder agar bisa menampilkan lalu menghilangkan teks
    placeholder_text = st.empty()
    placeholder_text.info("Sedang memproses data... Mohon tunggu beberapa detik.")

    # Buat jeda 1 detik
    time.sleep(1)

    # Kosongkan placeholder lalu ganti tulisan
    placeholder_text.empty()
    st.write("Silahkan klik dibawah untuk memprediksi!")

    # Normalisasi dengan scaler
    feature_cols = df.columns.drop('Date')
    scaler = load_scaler()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df

def predict_future_prices(model, df, num_days=1, window_size=30):
    """
    Memprediksi harga ke depan (num_days) menggunakan LSTM.
    Default di sini adalah 1 hari (num_days=1).
    """
    feature_cols = df.columns.drop('Date')
    data = df[feature_cols].values
    input_data = data[-window_size:].copy()
    scaler = load_scaler()

    future_dates = []
    predicted_prices = []

    for i in range(num_days):
        X_input = np.expand_dims(input_data, axis=0)
        y_pred = model.predict(X_input, verbose=0)

        # Update Price (kolom 0)
        X_last = input_data[-1].copy()
        X_last[0] = y_pred[0][0]

        # Inverse transform
        inv_result = scaler.inverse_transform([X_last])
        predicted_price = inv_result[0, 0]
        predicted_prices.append(predicted_price)

        # Geser window
        new_input = np.roll(input_data, shift=-1, axis=0)
        new_input[-1] = X_last
        input_data = new_input

        # Tentukan tanggal (bertambah satu hari)
        next_date = df["Date"].max() + timedelta(days=1)
        future_dates.append(next_date)

    return pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices})

# ==================================================================
# HALAMAN: DASHBOARD
# ==================================================================
def show_dashboard():
    st.subheader("Dashboard: Actual vs Predicted (Data Uji)")
    st.write(
        """
        Halaman ini menampilkan grafik *Actual vs Predicted Price* dari file
        **test_predictions.csv** (data uji).
        """
    )
    try:
        df_test = pd.read_csv("test_predictions.csv")
        df_test["Date"] = pd.to_datetime(df_test["Date"])
        df_test.sort_values("Date", inplace=True)

        st.markdown("**Beberapa baris teratas**:")
        st.dataframe(df_test.head())

        # Perhitungan MAPE sederhana (opsional)
        if "Actual Price" in df_test.columns and "Predicted Price" in df_test.columns:
            actual_vals = df_test["Actual Price"].values
            predicted_vals = df_test["Predicted Price"].values
            mape_val = np.mean(np.abs((actual_vals - predicted_vals) / actual_vals)) * 100
            st.write(f"**MAPE (Data Uji)**: {mape_val:.2f}%")

        # Plot Actual vs Predicted (jika ada kedua kolom)
        import plotly.express as px
        fig = px.line(df_test, x="Date", y=["Actual Price","Predicted Price"],
                      title="Perbandingan Actual vs Predicted Price (Test Data)")
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("File 'test_predictions.csv' tidak ditemukan. Pastikan file tersedia di direktori yang sama.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# ==================================================================
# HALAMAN: PREDIKSI
# ==================================================================
def show_prediction_page():
    st.subheader("Halaman Prediksi")
    # ------------ 1) Hilangkan st.info() default -------------
    

    st.markdown("""
    Halaman ini memungkinkan Anda mengunggah data historis minyak mentah **WTI**, 
    lalu menghasilkan prediksi harga untuk 1 hari ke depan.
    """)

    # -------------------------------------
    # Petunjuk Lengkap Format CSV
    # -------------------------------------
    st.markdown("""
    **Petunjuk Lengkap Format CSV:**

    - Wajib memiliki kolom: `Date`, `Price`, `Open`, `High`, `Low`, `Vol.`, `Change %`
    - Minimal data sebanyak 30 baris
    - Tanggal dengan format **MM/DD/YYYY**
    - Nilai volume (Vol.) dapat berupa angka dengan sufiks `K` atau `M`.
    - Kolom `Change %` adalah persentase perubahan, misalnya `1.2` untuk `1.2%`.
    """)

    st.markdown("""
    **Dapatkan data historis dari website berikut**:
    [Investing.com - Crude Oil Historical Data](https://www.investing.com/commodities/crude-oil-historical-data)
    """)

    uploaded_file = st.file_uploader(
        "📂 Unggah file CSV Anda di sini",
        type=["csv"],
        help="Format kolom harus mencakup Date, Price, Open, High, Low, Vol., dan Change %"
    )

    if not uploaded_file:
        st.warning("**Belum ada file yang diunggah. Silakan unggah file CSV** di atas.")
        st.stop()

    st.success("✅ File berhasil diunggah!")
    df = pd.read_csv(uploaded_file)

    # Validasi jumlah data
    if df.shape[0] < 30:
        st.error("❌ Data kurang dari 30 baris! Model LSTM membutuhkan minimal 30 baris data historis.")
        st.stop()

    # Info singkat data
    num_rows, num_cols = df.shape
    st.info("Ringkasan singkat data Anda:")
    st.write(f"- **Jumlah baris**: {num_rows}")
    st.write(f"- **Jumlah kolom**: {num_cols}")
    try:
        min_date = pd.to_datetime(df["Date"]).min().strftime('%Y-%m-%d')
        max_date = pd.to_datetime(df["Date"]).max().strftime('%Y-%m-%d')
        st.write(f"- **Range Tanggal**: {min_date} s/d {max_date}")
    except:
        st.write("- **Range Tanggal**: Tidak dapat dibaca (pastikan format kolom Date benar)")

    with st.expander("Lihat beberapa data teratas (head)", expanded=True):
        st.dataframe(df.head())

    # Preprocessing data
    df_cleaned = preprocess_raw_data(df)

    # Load model terlebih dahulu
    model = load_trained_model()

    # Tombol Prediksi
    if st.button("🚀 Prediksi dari Data CSV"):
        progress_bar = st.progress(0)
        for percent_complete in range(50):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)

        predictions_df = predict_future_prices(model, df_cleaned, num_days=1)
        progress_bar.progress(100)

        st.success("Prediksi berhasil dilakukan!")

        # Ambil nilai prediksi (1 hari)
        predicted_price_val = predictions_df["Predicted Price"].iloc[0]
        predicted_date_val = predictions_df["Date"].iloc[0]

        # Besarkan teks hasil prediksi, contoh: <h2>
        st.markdown(
            f"""
            <h2 style='text-align: left;'>
            Prediksi Harga pada {predicted_date_val.date()}: ${predicted_price_val:,.2f}
            </h2>
            """, 
            unsafe_allow_html=True
        )

        # =======================
        #  GRAFIK GABUNGAN PLOTLY
        # =======================
        scaler = load_scaler()
        feature_cols = df_cleaned.columns.drop('Date')
        df_unscaled = df_cleaned.copy()
        df_unscaled[feature_cols] = scaler.inverse_transform(df_cleaned[feature_cols])

        last_n_days = 7
        df_plot = df_unscaled[["Date", "Price"]].tail(last_n_days).copy()

        new_row = pd.DataFrame({
            "Date": [predicted_date_val],
            "Price": [predicted_price_val]
        })
        df_plot = pd.concat([df_plot, new_row], ignore_index=True)

        fig_plotly = px.line(
            df_plot, 
            x="Date", 
            y="Price", 
            markers=True,
            title="Harga Minyak Mentah WTI (7 Hari Terakhir + 1 Hari Prediksi)"
        )
        fig_plotly.update_layout(xaxis_title="Tanggal", yaxis_title="Harga (USD)")
        st.plotly_chart(fig_plotly, use_container_width=True)

        # Unduh hasil prediksi
        csv_data = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Unduh Hasil Prediksi (CSV)",
            csv_data,
            "future_predictions_1_day.csv",
            "text/csv",
            help="Klik untuk mengunduh hasil prediksi dalam format CSV."
        )

# ==================================================================
# MAIN
# ==================================================================
def main():
    if st.session_state["page"] == "Dashboard":
        show_dashboard()
    else:
        show_prediction_page()

if __name__ == "__main__":
    main()

# -------------------------------------
# Footer
# -------------------------------------
st.markdown("---")
st.write("📌 **Catatan:** Model ini dibuat untuk tujuan edukasi dan tidak dapat dijadikan sebagai saran keuangan.")
st.write("📧 **Kontak**: m.dzahwan@gmail.com | [LinkedIn](https://www.linkedin.com/in/mochammad-dzahwan/?originalSubdomain=id)")
