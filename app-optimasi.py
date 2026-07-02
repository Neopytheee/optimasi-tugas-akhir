# -*- coding: utf-8 -*-
"""
File: app.py
Deskripsi: Aplikasi Web Streamlit Tunggal untuk Sistem Analisis Kelayakan Kredit.
           Mengintegrasikan UI Form, Live Min-Max Scaling, Komputasi Jarak Euclidean LVQ-PSO,
           serta Manajemen Penyimpanan Riwayat menggunakan SQLite.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

# ==========================================
# 1. INISIALISASI BASIS DATA (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS riwayat_nasabah (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama TEXT,
            anak INTEGER,
            pendapatan REAL,
            umur REAL,
            masa_kerja REAL,
            keluarga INTEGER,
            status_kelayakan TEXT,
            waktu_evaluasi TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 2. KONFIGURASI HALAMAN UTAMA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Sistem Kredit LVQ-PSO",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Sistem Analisis Kelayakan Kredit Nasabah")
st.markdown("### *Hybrid Intelligence Engine* Berbasis Learning Vector Quantization (LVQ) & Particle Swarm Optimization (PSO)")
st.write("---")

# Membuat layout kolom: Kiri untuk Form Input, Kanan untuk Hasil & Riwayat
col_form, col_hasil = st.columns([1, 1.2])

# ==========================================
# 3. AREA KIRI: FORM DATA ENTRY
# ==========================================
with col_form:
    st.header("📋 Form Evaluasi Nasabah Baru")
    
    with st.form(key='form_nasabah', clear_on_submit=True):
        nama = st.text_input("Nama Lengkap Nasabah", placeholder="Contoh: Ahmad Fauzi")
        anak = st.number_input("Jumlah Anak / Tanggungan (CNT_CHILDREN)", min_value=0, max_value=20, step=1, value=0)
        pendapatan = st.number_input("Total Pendapatan Tahunan (AMT_INCOME_TOTAL)", min_value=0.0, step=1000.0, value=50000.0)
        umur = st.number_input("Usia Nasabah (Tahun)", min_value=18, max_value=100, step=1, value=30)
        kerja = st.number_input("Masa Kerja Aktif (Tahun)", min_value=0, max_value=60, step=1, value=5)
        keluarga = st.number_input("Jumlah Anggota Keluarga (CNT_FAM_MEMBERS)", min_value=1, max_value=25, step=1, value=2)
        
        submit_button = st.form_submit_button(label="🧬 Proses Kelayakan Kredit")

# ==========================================
# 4. AREA KANAN: CORE PROCESSING ENGINE & RIWAYAT
# ==========================================
with col_hasil:
    st.header("📊 Hasil Analisis Keputusan")
    
    if submit_button:
        if not nama:
            st.error("⚠️ Mohon isi Nama Nasabah terlebih dahulu sebelum memproses.")
        else:
            # --- A. PENYESUAIAN PARAMETER (SINKRON DATASET) ---
            # Mengubah format input tahun menjadi format hari negatif sesuai dataset asli
            days_birth = umur * -365.25
            days_employed = kerja * -365.25
            
            vektor_mentah = np.array([anak, pendapatan, days_birth, days_employed, keluarga])
            
            # --- B. LIVE MIN-MAX SCALING ---
            min_vals = np.array([0.0, 27000.0, -25170.0, -17531.0, 1.0]) 
            max_vals = np.array([19.0, 1575000.0, -7489.0, 365243.0, 20.0])
            
            vektor_input_norm = (vektor_mentah - min_vals) / (max_vals - min_vals)
            vektor_input_norm = np.clip(vektor_input_norm, 0.0, 1.0)
            
            # --- C. MATRIKS BOBOT PROTOTIPE (G-BEST PSO) ---
            # Masukkan hasil koordinat partikel gbest terbaik dari Google Colab Anda di sini
            bobot_lvq_pso = np.array([
                [0.0521, 0.1245, 0.5421, 0.0412, 0.0833],  # Centroid Kelas 0: LAYAK
                [0.1152, 0.0421, 0.7812, 0.0105, 0.1667]   # Centroid Kelas 1: TIDAK LAYAK
            ])
            
            # --- D. KOMPUTASI DISTANCE LVQ ---
            jarak_ke_layak = np.linalg.norm(vektor_input_norm - bobot_lvq_pso[0])
            jarak_ke_tidak_layak = np.linalg.norm(vektor_input_norm - bobot_lvq_pso[1])
            
            # --- E. PENENTUAN PEMENANG KELAS ---
            if jarak_ke_layak < jarak_ke_tidak_layak:
                status_kelayakan = "LAYAK (GOOD CLIENT)"
                warna_box = "success"
            else:
                status_kelayakan = "TIDAK LAYAK (BAD CLIENT)"
                warna_box = "error"
            
            # --- F. CETAK KEPUTUSAN KE VISUAL UI ---
            st.markdown("#### Status Keputusan:")
            if warna_box == "success":
                st.success(f"🎉 **{status_kelayakan}**")
            else:
                st.error(f"❌ **{status_kelayakan}**")
                
            # Menampilkan detail perhitungan spasial
            with st.expander("🔍 Lihat Detail Nilai Kedekatan Ruang Vektor (Euclidean Distance)"):
                st.write(f"• Jarak Euclidean ke Centroid Kelas LAYAK: `{jarak_ke_layak:.6f}`")
                st.write(f"• Jarak Euclidean ke Centroid Kelas TIDAK LAYAK: `{jarak_ke_tidak_layak:.6f}`")
                st.caption("Keputusan diambil berdasarkan unit pemenang dengan nilai jarak spasial terpendek (Minimum Distance).")

            # --- G. SIMPAN DATA KE DATABASE SQLITE ---
            waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO riwayat_nasabah (nama, anak, pendapatan, umur, masa_kerja, keluarga, status_kelayakan, waktu_evaluasi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (nama, anak, pendapatan, umur, kerja, keluarga, status_kelayakan, waktu_sekarang))
            conn.commit()
            conn.close()
            st.toast(f"Data transaksi atas nama {nama} berhasil diamankan ke database!", icon='💾')

    # ==========================================
    # 5. MENAMPILKAN RIWAYAT LOG DATABASE
    # ==========================================
    st.write("---")
    st.subheader("📜 Log Riwayat Evaluasi Kredit")
    
    conn = sqlite3.connect('database.db')
    # Mengambil data terbaru ditaruh di urutan paling atas
    df_riwayat = pd.read_sql_query("SELECT nama, anak, pendapatan, umur, masa_kerja, keluarga, status_kelayakan, waktu_evaluasi FROM riwayat_nasabah ORDER BY id DESC", conn)
    conn.close()
    
    if not df_riwayat.empty:
        # Mempercantik nama kolom saat dirender di tabel Streamlit
        df_riwayat.columns = ['Nama Nasabah', 'Anak', 'Pendapatan (Th)', 'Usia', 'Masa Kerja', 'Anggota Keluarga', 'Keputusan Sistem', 'Waktu Evaluasi']
        st.dataframe(df_riwayat, use_container_width=True)
    else:
        st.info("Belum ada riwayat transaksi data nasabah di dalam database.")
