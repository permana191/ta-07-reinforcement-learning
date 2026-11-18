# TA-06 Reinforcement Learning: Smart EV Charging Scheduler (Q-Learning)

- **Nama:** Sigit Miraj Permana
- **NIM:** 301240037
- **Mata Praktikum:** Pembelajaran Mesin (Modul 07 — Reinforcement Learning)
- **Dosen Pengampu:** Mohammad Bayu Anggara, S.Kom., M.Kom.

## Struktur Direktori Proyek
```text
TA06_RL_Smart_EV_Charging_301240037/
├── app.py                                      # Aplikasi Web Dashboard Interaktif (Flask)
├── train_model.py                              # Skrip Utama Pelatihan & Eksperimen (Langkah 1-5)
├── requirements.txt                            # Daftar dependensi library Python
├── README.md                                   # Panduan instalasi, arsitektur, dan penjelasan
├── src/
│   ├── __init__.py
│   ├── environment.py                          # Simulator Lingkungan Microgrid PV-EV (27 States)
│   └── q_learning_agent.py                     # Algoritma Tabular Q-Learning & Persamaan Bellman
├── notebooks/
│   └── TA06_QLearning_Smart_EV_Charging.ipynb  # Jupyter Notebook lengkap dengan narasi analisis
├── templates/
│   └── index.html                              # Antarmuka Web Dashboard Putih-Biru
├── static/
│   ├── css/style.css                           # Stylesheet terpisah
│   └── js/main.js                              # Logika AJAX simulasi interaktif
└── outputs/
    ├── models/                                 # Q-Table terlatih (.npy), Policy (.csv), Metrik (.json)
    └── plots/                                  # 5 Grafik resolusi tinggi siap masuk laporan PDF/DOC
```

## Cara Menjalankan Proyek (Windows 11 / VS Code / PowerShell)

1. **Install Dependensi:**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Jalankan Pelatihan Model & Generate Grafik (Langkah 1–5):**
   ```powershell
   python train_model.py
   ```
3. **Jalankan Dashboard Web Interaktif:**
   ```powershell
   python app.py
   ```
   Buka browser di alamat `http://127.0.0.1:5000`.
4. **Buka Jupyter Notebook (`.ipynb`):**
   Buka file `notebooks/TA06_QLearning_Smart_EV_Charging.ipynb` di VS Code atau Google Colab.
