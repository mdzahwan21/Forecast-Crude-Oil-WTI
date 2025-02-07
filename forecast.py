import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import io
from datetime import datetime, timedelta
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
        2. Klik tombol **Prediksi** untuk memproses data Anda.
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


    # Normalisasi dengan scaler
    feature_cols = df.columns.drop('Date')
    scaler = load_scaler()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df

def predict_future_prices(model, df, num_days=3, window_size=30):
    """Memprediksi harga ke depan (num_days) menggunakan LSTM."""
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

        # Tentukan tanggal (next day)
        if i == 0:
            next_date = df["Date"].max() + timedelta(days=1)
        else:
            next_date = future_dates[-1] + timedelta(days=1)
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

        # Plot Actual vs Predicted
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_test["Date"], df_test["Actual Price"], label="Actual Price", linestyle='solid')
        ax.plot(df_test["Date"], df_test["Predicted Price"], label="Predicted Price", linestyle='dashed')
        ax.set_title("Perbandingan Actual vs Predicted Price (Test Data)")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (USD)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        st.pyplot(fig)

    except FileNotFoundError:
        st.error("File 'test_predictions.csv' tidak ditemukan. Pastikan file tersedia di direktori yang sama.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# ==================================================================
# HALAMAN: PREDIKSI
# ==================================================================
def show_prediction_page():
    st.subheader("Halaman Prediksi")
    st.markdown("""
    Halaman ini memungkinkan Anda mengunggah data historis minyak mentah **WTI**, 
    lalu menghasilkan prediksi harga untuk 3 hari ke depan.
    """)

    # -------------------------------------
    # 4. Petunjuk Lengkap Format CSV
    # -------------------------------------
    st.markdown("""
    **Petunjuk Lengkap Format CSV:**

    - Wajib memiliki kolom: `Date`, `Price`, `Open`, `High`, `Low`, `Vol.`, `Change %`
    - Minimal data sebanyak 30 baris
    - Tanggal dengan format **MM/DD/YYYY**
    - Nilai volume (Vol.) dapat berupa angka dengan sufiks `K` atau `M`.
    - Kolom `Change %` adalah persentase perubahan, misalnya `1.2` untuk `1.2%`.
    - Jika Anda belum memiliki data, dapatkan data dari link dibawah , lalu coba unggah kembali.
    """)

    # -------------------------------------
    # 5. Dapatkan Data Historis
    # -------------------------------------
    st.markdown("""
    **Dapatkan data historis dari website berikut**:
    [Investing.com - Crude Oil Historical Data](https://www.investing.com/commodities/crude-oil-historical-data)
    """)

    # -------------------------------------
    # 6. Upload File CSV
    # -------------------------------------
    uploaded_file = st.file_uploader(
        "📂 Unggah file CSV Anda di sini",
        type=["csv"],
        help="Format kolom harus mencakup Date, Price, Open, High, Low, Vol., dan Change %"
    )

    # -------------------------------------
    # 7. Jika File Belum Diunggah
    # -------------------------------------
    if not uploaded_file:
        st.warning("**Belum ada file yang diunggah. Silakan unggah file CSV** di atas untuk memulai prediksi.")
        st.stop()

    # Baca data
    st.success("✅ File berhasil diunggah!")
    df = pd.read_csv(uploaded_file)

    # 🚨 Tambahkan validasi jumlah data sebelum proses lebih lanjut
    if df.shape[0] < 22:
        st.error("❌ Data kurang dari 30 baris! Model LSTM membutuhkan minimal 30 baris data historis untuk prediksi yang akurat.")
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

    # Preprocessing
    st.info("Sedang memproses data... Mohon tunggu beberapa detik.")
    df_cleaned = preprocess_raw_data(df)

    # Tombol Prediksi
    if st.button("🚀 Prediksi dari Data CSV"):
        with st.spinner("Menghasilkan prediksi..."):
            model = load_trained_model()
            predictions_df = predict_future_prices(model, df_cleaned)
        st.success("Prediksi berhasil dilakukan!")
        st.balloons()

        st.markdown("### 📉 Hasil Prediksi untuk 3 Hari ke Depan")
        st.dataframe(predictions_df.style.format({"Predicted Price": "${:,.2f}"}))

        # Visualisasi hasil
        st.markdown("### 📈 Grafik Prediksi")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(predictions_df["Date"], predictions_df["Predicted Price"], 
                marker='o', linestyle='dashed', color='blue', label="Predicted Price")
        ax.set_title("Prediksi Harga Minyak Mentah WTI (3 Hari ke Depan)")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (USD)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        st.pyplot(fig)

        # Unduh hasil prediksi
        csv_data = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Unduh Hasil Prediksi",
            csv_data,
            "future_predictions_3_days.csv",
            "text/csv",
            help="Klik untuk mengunduh hasil prediksi dalam format CSV."
        )

# ==================================================================
# MAIN
# ==================================================================
def main():
    # Cek halaman aktif di session_state
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





# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import io

# # -------------------------------------
# # 1. Konfigurasi Halaman
# # -------------------------------------
# st.set_page_config(
#     page_title="Prediksi Harga Minyak Mentah WTI",
#     page_icon="⛽",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # -------------------------------------
# # 2. Heading Aplikasi
# # -------------------------------------
# st.markdown(
#     """
#     <h1 style='text-align: center; margin-bottom: 0px;'>📈 Prediksi Harga Minyak Mentah WTI</h1>
#     <p style='text-align: center; color: grey; margin-top: 0px;'>
#     Menggunakan model LSTM berbasis data historis
#     </p>
#     <hr>
#     """,
#     unsafe_allow_html=True
# )

# # -------------------------------------
# # 3. Sidebar
# # -------------------------------------
# st.sidebar.title("⚙️ Pengaturan & Tentang Aplikasi")
# with st.sidebar.expander("❓ Bagaimana cara kerja aplikasi ini?", expanded=True):
#     st.write(
#         """
#         Aplikasi ini menggunakan **model LSTM** yang telah dilatih untuk memprediksi harga minyak mentah WTI 
#         berdasarkan rangkaian waktu data historis.\n
#         **Langkah-langkah** penggunaannya:
#         1. Unggah file CSV berisi data historis minimal sebanyak 30 baris (Date, Price, Open, High, Low, Vol., Change %).
#         2. Klik tombol **Prediksi** untuk memproses data Anda.
#         3. Lihat hasil prediksi dan unduh jika diperlukan.
#         """
#     )

# # -------------------------------------
# # 4. Petunjuk Lengkap Format CSV
# # -------------------------------------
# st.markdown("""
# **Petunjuk Lengkap Format CSV:**

# - Wajib memiliki kolom: `Date`, `Price`, `Open`, `High`, `Low`, `Vol.`, `Change %`
# - Minimal data sebanyak 30 baris
# - Tanggal dengan format **MM/DD/YYYY**
# - Nilai volume (Vol.) dapat berupa angka dengan sufiks `K` atau `M`.
# - Kolom `Change %` adalah persentase perubahan, misalnya `1.2` untuk `1.2%`.
# - Jika Anda belum memiliki data, dapatkan data dari link dibawah , lalu coba unggah kembali.
# """)

# # -------------------------------------
# # 5. Dapatkan Data Historis
# # -------------------------------------
# st.markdown("""
# **Dapatkan data historis dari website berikut**:
# [Investing.com - Crude Oil Historical Data](https://www.investing.com/commodities/crude-oil-historical-data)
# """)


# # -------------------------------------
# # 6. Upload File CSV
# # -------------------------------------
# uploaded_file = st.file_uploader(
#     "📂 Unggah file CSV Anda di sini",
#     type=["csv"],
#     help="Format kolom harus mencakup Date, Price, Open, High, Low, Vol., dan Change %"
# )

# # -------------------------------------
# # 7. Jika File Belum Diunggah
# # -------------------------------------
# if not uploaded_file:
#     st.warning("**Belum ada file yang diunggah. Silakan unggah file CSV** di atas untuk memulai prediksi.")
#     st.stop()

# # -------------------------------------
# # 8. Fungsi Load Model & Scaler
# # -------------------------------------
# @st.cache_resource
# def load_trained_model():
#     return load_model("lstm_wti_model.h5", compile=False)

# @st.cache_resource
# def load_scaler():
#     return joblib.load("minmax_scaler.pkl")

# # -------------------------------------
# # 9. Fungsi Preprocessing Data
# # -------------------------------------
# def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
#     df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
#     df.sort_values('Date', inplace=True)
#     df["Change %"] = df["Change %"].str.replace('%','', regex=True).astype(float)

#     def convert_volume(vol_str):
#         if isinstance(vol_str, str):
#             vol_str = vol_str.replace(',', '')
#             if 'K' in vol_str:
#                 return float(vol_str.replace('K','')) * 1000
#             elif 'M' in vol_str:
#                 return float(vol_str.replace('M','')) * 1000000
#             return float(vol_str)
#         return vol_str

#     df['Vol.'] = df['Vol.'].apply(convert_volume)

#     # Set Date sebagai index dan reindex agar berurutan harian
#     df.set_index('Date', inplace=True)
#     df = df.reindex(
#         pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
#     )
#     df.interpolate(method='linear', inplace=True)
#     df.reset_index(inplace=True)
#     df.rename(columns={'index': 'Date'}, inplace=True)

#     # Normalisasi fitur
#     feature_cols = df.columns.drop('Date')
#     scaler = load_scaler()
#     df[feature_cols] = scaler.transform(df[feature_cols])
#     return df

# # -------------------------------------
# # 10. Fungsi Prediksi
# # -------------------------------------
# def predict_future_prices(model, df, num_days=3, window_size=30):
#     feature_cols = df.columns.drop('Date')
#     data = df[feature_cols].values
#     input_data = data[-window_size:].copy()

#     future_dates = [df['Date'].max() + timedelta(days=i) for i in range(1, num_days + 1)]
#     predicted_prices = []

#     for i in range(num_days):
#         X_input = np.expand_dims(input_data, axis=0)
#         y_pred = model.predict(X_input)

#         # Update Price (asumsi kolom 0 adalah Price)
#         X_last = input_data[-1].copy()
#         X_last[0] = y_pred[0][0]

#         # Inverse transform
#         inv_result = load_scaler().inverse_transform([X_last])
#         predicted_price = inv_result[0, 0]
#         predicted_prices.append(predicted_price)

#         # Geser window
#         new_input = np.roll(input_data, shift=-1, axis=0)
#         new_input[-1] = X_last
#         input_data = new_input

#     return pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices})

# # -------------------------------------
# # 11. Jika File Sudah Diunggah
# # -------------------------------------
# st.success("✅ File berhasil diunggah!")
# df = pd.read_csv(uploaded_file)

# # Info singkat data
# st.info("Ringkasan singkat data Anda:")
# num_rows, num_cols = df.shape
# st.write(f"- **Jumlah baris**: {num_rows}")
# st.write(f"- **Jumlah kolom**: {num_cols}")

# try:
#     min_date = pd.to_datetime(df["Date"]).min().strftime('%Y-%m-%d')
#     max_date = pd.to_datetime(df["Date"]).max().strftime('%Y-%m-%d')
#     st.write(f"- **Range Tanggal**: {min_date} s/d {max_date}")
# except:
#     st.write("- **Range Tanggal**: Tidak dapat dibaca (pastikan format kolom Date benar)")

# with st.expander("Lihat beberapa data teratas (head)", expanded=True):
#     st.dataframe(df.head())

# # -------------------------------------
# # 12. Preprocessing Data
# # -------------------------------------
# st.info("Sedang memproses data... Mohon tunggu beberapa detik.")
# df_cleaned = preprocess_raw_data(df)

# # -------------------------------------
# # 13. Tombol Prediksi
# # -------------------------------------
# if st.button("🚀 Prediksi dari Data CSV"):
#     with st.spinner("Menghasilkan prediksi..."):
#         model = load_trained_model()
#         predictions_df = predict_future_prices(model, df_cleaned)
#     st.success("Prediksi berhasil dilakukan!")
#     st.balloons()  # efek balon opsional

#     st.markdown("### 📉 Hasil Prediksi untuk 3 Hari ke Depan")
#     st.dataframe(predictions_df.style.format({"Predicted Price": "${:,.2f}"}))

#     # -------------------------------------
#     # Visualisasi
#     # -------------------------------------
#     st.markdown("### 📈 Grafik Prediksi")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.plot(predictions_df["Date"], predictions_df["Predicted Price"], 
#             marker='o', linestyle='dashed', color='blue', label="Predicted Price")
#     ax.set_title("Prediksi Harga Minyak Mentah WTI (3 Hari ke Depan)")
#     ax.set_xlabel("Tanggal")
#     ax.set_ylabel("Harga (USD)")
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.grid()
#     st.pyplot(fig)

#     # -------------------------------------
#     # Tombol Unduh Hasil Prediksi
#     # -------------------------------------
#     csv_data = predictions_df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         "📥 Unduh Hasil Prediksi",
#         csv_data,
#         "future_predictions_3_days.csv",
#         "text/csv",
#         help="Klik untuk mengunduh hasil prediksi dalam format CSV."
#     )

# # -------------------------------------
# # 14. Footer
# # -------------------------------------
# st.markdown("---")
# st.write("📌 **Catatan:** Model ini dibuat untuk tujuan edukasi dan tidak dapat dijadikan sebagai saran keuangan.")
# st.write("📧 **Kontak**: m.dzahwan@gmail.com | [LinkedIn](https://www.linkedin.com/in/mochammad-dzahwan/?originalSubdomain=id)")







# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt

# # Streamlit UI
# st.title("Prediksi Harga Minyak Mentah WTI")

# # Muat model dan scaler
# @st.cache_resource
# def load_trained_model():
#     return load_model("lstm_wti_model.h5", compile=False)

# @st.cache_resource
# def load_scaler():
#     return joblib.load("minmax_scaler.pkl")

# # Fungsi untuk membersihkan dan memproses dataset mentah
# def preprocess_raw_data(df):
#     df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
#     df.sort_values('Date', inplace=True)
#     df["Change %"] = df["Change %"].str.replace('%','', regex=True).astype(float)
    
#     def convert_volume(vol_str):
#         if isinstance(vol_str, str):
#             vol_str = vol_str.replace(',', '')
#             if 'K' in vol_str:
#                 return float(vol_str.replace('K','')) * 1000
#             elif 'M' in vol_str:
#                 return float(vol_str.replace('M','')) * 1000000
#             return float(vol_str)
#         return vol_str
    
#     df['Vol.'] = df['Vol.'].apply(convert_volume)
    
#     df.set_index('Date', inplace=True)
#     df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='D'))
#     df.interpolate(method='linear', inplace=True)
#     df.reset_index(inplace=True)
#     df.rename(columns={'index': 'Date'}, inplace=True)
    
#     feature_cols = df.columns.drop('Date')
#     scaler = load_scaler()
#     df[feature_cols] = scaler.transform(df[feature_cols])
#     return df

# # Fungsi untuk melakukan prediksi
# def predict_future_prices(model, df, num_days=3, window_size=30):
#     feature_cols = df.columns.drop('Date')
#     data = df[feature_cols].values
#     input_data = data[-window_size:].copy()
#     future_dates = [df['Date'].max() + timedelta(days=i) for i in range(1, num_days + 1)]
#     predicted_prices = []
    
#     for i in range(num_days):
#         X_input = np.expand_dims(input_data, axis=0)
#         y_pred = model.predict(X_input)
#         X_last = input_data[-1].copy()
#         X_last[0] = y_pred[0][0]
#         inv_result = load_scaler().inverse_transform([X_last])
#         predicted_price = inv_result[0, 0]
#         predicted_prices.append(predicted_price)
#         new_input = np.roll(input_data, shift=-1, axis=0)
#         new_input[-1] = X_last
#         input_data = new_input
    
#     return pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices})

# # Fungsi untuk prediksi manual
# def predict_manual_price(model, scaler, manual_data):
#     feature_cols = ["Price","Open", "High", "Low", "Vol.", "Change %"]
#     scaled_input = scaler.transform(manual_data[feature_cols])
#     scaled_input = np.expand_dims(scaled_input, axis=0)
#     y_pred = model.predict(scaled_input)
    
#     X_last = manual_data.iloc[0].copy()
#     X_last["Open"] = y_pred[0][0]
#     inv_result = scaler.inverse_transform([X_last.values])
#     predicted_price = inv_result[0, 0]
    
#     return predicted_price

# # Input Form untuk Data Manual
# st.sidebar.header("Masukkan Data Manual")
# price = st.sidebar.number_input("Price", value=72.0)
# open_price = st.sidebar.number_input("Open Price", value=70.0)
# high_price = st.sidebar.number_input("High Price", value=75.0)
# low_price = st.sidebar.number_input("Low Price", value=65.0)
# volume = st.sidebar.number_input("Volume", value=1000000.0)
# change_percent = st.sidebar.number_input("Change %", value=0.5)

# if st.sidebar.button("Prediksi Harga untuk 1 Hari ke Depan"):
#     manual_data = pd.DataFrame({
#         "Price": [price],
#         "Open": [open_price],
#         "High": [high_price],
#         "Low": [low_price],
#         "Vol.": [volume],
#         "Change %": [change_percent]
#     })
#     model = load_trained_model()
#     scaler = load_scaler()
#     predicted_price = predict_manual_price(model, scaler, manual_data)
#     st.sidebar.write(f"### Prediksi Harga untuk Besok: {predicted_price:.2f}")

# # Upload file CSV
# uploaded_file = st.file_uploader("Unggah file CSV dengan data mentah minyak mentah", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     df = preprocess_raw_data(df)
#     st.write("### Data yang Telah Dibersihkan:")
#     st.dataframe(df.head())
    
#     if st.button("Prediksi dari Data CSV"):
#         model = load_trained_model()
#         predictions_df = predict_future_prices(model, df)
        
#         st.write("### Hasil Prediksi untuk 3 Hari ke Depan:")
#         st.dataframe(predictions_df)
        
#         # Visualisasi hasil prediksi
#         plt.figure(figsize=(10, 5))
#         plt.plot(predictions_df["Date"], predictions_df["Predicted Price"], marker='o', linestyle='dashed', color='blue', label="Predicted Price")
#         plt.title("Prediksi Harga Minyak Mentah WTI")
#         plt.xlabel("Tanggal")
#         plt.ylabel("Harga Minyak")
#         plt.xticks(rotation=45)
#         plt.legend()
#         plt.grid()
#         st.pyplot(plt)
        
#         # Tombol unduh hasil prediksi
#         csv_data = predictions_df.to_csv(index=False).encode('utf-8')
#         st.download_button("Unduh Hasil Prediksi", csv_data, "future_predictions_3_days.csv", "text/csv")

