import numpy as np


class SmartEVChargingEnv:
    """
    Lingkungan Simulasi Microgrid PV-EV Berbasis State Diskrit (27 States x 3 Actions).

    Definisi State Space (S) -> Total 3 x 3 x 3 = 27 States:
    ---------------------------------------------------------
    1. Kondisi Energi Bersih / Net Power (net_power_idx):
       - 0: Defisit Daya      (Beban total melebihi suplai PV + Grid dasar)
       - 1: Surplus Tipis     (Surplus daya kecil, rawan overload jika fast charging)
       - 2: Surplus Melimpah  (Suplai PV tinggi, sangat aman untuk pengisian penuh)
    2. Level Baterai Penyimpanan Stasiun / Battery SOC (soc_idx):
       - 0: Rendah / Kritis   (< 35%)
       - 1: Sedang / Normal   (35% - 70%)
       - 2: Tinggi / Penuh    (> 70%)
    3. Tarif Listrik Dinamis / Time-of-Use Tariff (tariff_idx):
       - 0: Murah (Off-Peak)  (00:00 - 06:00 & 11:00 - 15:00 saat puncak matahari)
       - 1: Normal (Mid-Peak) (07:00 - 10:00 & 16:00 - 17:00)
       - 2: Mahal (Peak Load) (18:00 - 23:00 beban puncak perumahan & bisnis)

    Definisi Action Space (A) -> Total 3 Actions:
    ---------------------------------------------
    - 0: Standby / Tunda Charging (Menahan pengisian EV, fokus mengisi baterai stasiun)
    - 1: Eco Charging (PV + Baterai) (Pengisian daya EV moderat 11 kW berbasis energi hijau)
    - 2: Fast Charging (PV + Grid) (Pengisian daya EV penuh 22 kW kombinasi PV & Grid)
    """

    def __init__(self, max_steps: int = 24, seed: int = None):
        """
        Inisialisasi parameter fisik dan ruang observasi lingkungan microgrid.

        Parameter:
            max_steps (int): Durasi satu episode simulasi (default 24 jam operasional).
            seed (int): Random seed untuk reproduktibilitas eksperimen.
        """
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Dimensi ruang keadaan dan aksi
        self.n_power_levels = 3
        self.n_soc_levels = 3
        self.n_tariff_levels = 3
        self.n_states = self.n_power_levels * self.n_soc_levels * self.n_tariff_levels  # 27
        self.n_actions = 3

        # Label deskriptif untuk keperluan interpretasi dan visualisasi
        self.power_labels = ["Defisit", "Surplus Tipis", "Surplus Melimpah"]
        self.soc_labels = ["SOC Rendah", "SOC Sedang", "SOC Tinggi"]
        self.tariff_labels = ["Tarif Murah", "Tarif Normal", "Tarif Mahal"]
        self.action_labels = [
            "0: Standby / Tunda",
            "1: Eco Charging (PV+Bat)",
            "2: Fast Charging (PV+Grid)"
        ]

        # Variabel internal simulasi
        self.current_hour = 0
        self.step_count = 0
        self.battery_soc = 50.0  # Persentase awal baterai stasiun (0 - 100%)
        self.state_tuple = (1, 1, 0)

    def encode_state(self, power_idx: int, soc_idx: int, tariff_idx: int) -> int:
        """
        Mengonversi koordinat 3 dimensi (power, soc, tariff) menjadi indeks tunggal
        0..26 agar dapat dipetakan langsung ke baris Q-Table.
        Rumus: state_id = (power_idx * 9) + (soc_idx * 3) + tariff_idx
        """
        return (power_idx * 9) + (soc_idx * 3) + tariff_idx

    def decode_state(self, state_id: int) -> tuple:
        """
        Mengembalikan indeks state (0..26) ke bentuk tuple (power_idx, soc_idx, tariff_idx).
        """
        power_idx = state_id // 9
        rem = state_id % 9
        soc_idx = rem // 3
        tariff_idx = rem % 3
        return (power_idx, soc_idx, tariff_idx)

    def get_state_description(self, state_id: int) -> str:
        """
        Menghasilkan label teks yang mudah dibaca manusia untuk setiap state_id.
        """
        p, s, t = self.decode_state(state_id)
        return f"{self.power_labels[p]} | {self.soc_labels[s]} | {self.tariff_labels[t]}"

    def _simulate_microgrid_physics(self, hour: int):
        """
        Menghitung kondisi fisik microgrid pada jam tertentu (0-23) menggunakan
        model kurva radiasi matahari Gaussian dan profil beban perkotaan.
        """
        # 1. Kurva Pembangkitan Surya (PV): Puncak pada jam 12:00 - 13:00 (maks ~48 kW)
        solar_mean = 48.0 * np.exp(-0.5 * ((hour - 12.5) / 3.2) ** 2)
        cloud_factor = self.rng.uniform(0.70, 1.10)  # Fluktuasi awan stokastik
        solar_kw = float(np.clip(solar_mean * cloud_factor, 0.0, 50.0))

        # 2. Profil Beban Dasar (Rumah Tangga + Bisnis + Antrean EV dasar)
        if 18 <= hour <= 22:
            # Jam beban puncak malam hari
            base_demand = self.rng.uniform(42.0, 62.0)
            tariff_idx = 2  # Tarif Mahal
        elif (7 <= hour <= 10) or (16 <= hour <= 17):
            # Jam transisi pagi dan sore
            base_demand = self.rng.uniform(28.0, 44.0)
            tariff_idx = 1  # Tarif Normal
        else:
            # Jam tengah hari (PV melimpah) atau dini hari (beban rendah)
            base_demand = self.rng.uniform(16.0, 32.0)
            tariff_idx = 0  # Tarif Murah

        # 3. Suplai Dasar Grid & Net Available Power (kW)
        grid_base_allowance = 22.0
        net_power_kw = (solar_kw + grid_base_allowance) - base_demand

        # Diskritisasi Net Power ke 3 kategori
        if net_power_kw <= 0.0:
            power_idx = 0  # Defisit
        elif net_power_kw <= 12.0:
            power_idx = 1  # Surplus Tipis
        else:
            power_idx = 2  # Surplus Melimpah

        # Diskritisasi Battery SOC (%) ke 3 kategori
        if self.battery_soc < 35.0:
            soc_idx = 0    # Rendah
        elif self.battery_soc <= 70.0:
            soc_idx = 1    # Sedang
        else:
            soc_idx = 2    # Tinggi

        return power_idx, soc_idx, tariff_idx, solar_kw, net_power_kw

    def reset(self):
        """
        Mereset simulasi ke awal episode (jam 06:00 pagi dengan SOC baterai acak).
        Mengembalikan state awal (int) dan dictionary informasi diagnostik.
        """
        self.current_hour = int(self.rng.integers(0, 24))
        self.step_count = 0
        self.battery_soc = float(self.rng.uniform(30.0, 75.0))

        p_idx, s_idx, t_idx, solar_kw, net_kw = self._simulate_microgrid_physics(self.current_hour)
        self.state_tuple = (p_idx, s_idx, t_idx)
        state_id = self.encode_state(p_idx, s_idx, t_idx)

        info = {
            "hour": self.current_hour,
            "solar_kw": round(solar_kw, 2),
            "net_power_kw": round(net_kw, 2),
            "battery_soc": round(self.battery_soc, 2)
        }
        return state_id, info

    def step(self, action: int):
        """
        Mengeksekusi satu langkah aksi agen pada lingkungan dan menghitung:
        1. Perubahan fisik Battery SOC (%)
        2. Reward numerik (r) berdasarkan aturan insentif & penalti
        3. Transisi ke state berikutnya (new_state)
        4. Status terminal episode (done)
        """
        assert action in [0, 1, 2], f"Aksi tidak valid: {action}"

        p_idx, s_idx, t_idx = self.state_tuple
        reward = 0.0
        status_note = ""

        # ======================================================================
        # LOGIKA TRANSISI FISIKA & REWARD FUNCTION (R)
        # ======================================================================
        if action == 0:
            # AKSI 0: STANDBY / TUNDA CHARGING
            # Baterai stasiun terisi perlahan jika ada surplus surya
            charge_gain = 6.0 if p_idx == 2 else (3.0 if p_idx == 1 else 1.0)
            self.battery_soc = min(100.0, self.battery_soc + charge_gain)

            if p_idx == 0 or (t_idx == 2 and s_idx == 0):
                # Tepat! Menunda charging saat defisit daya atau tarif puncak & baterai kritis
                reward = +12.0
                status_note = "Optimal Protection: Mencegah Grid Overload saat defisit/tarif mahal."
            elif p_idx == 2 and s_idx == 2 and t_idx == 0:
                # Kurang optimal: Energi melimpah, baterai penuh, tarif murah, tapi tidak melayani EV
                reward = -6.0
                status_note = "Opportunity Loss: Energi surya terbuang sia-sia saat baterai penuh."
            else:
                reward = +2.0
                status_note = "Safe Standby: Menjaga stabilitas cadangan baterai stasiun."

        elif action == 1:
            # AKSI 1: ECO CHARGING (PV + BATERAI) -> Beban 11 kW
            # Menggunakan energi surya dan sedikit cadangan baterai tanpa membebani grid mahal
            soc_drain = 4.0 if p_idx == 2 else (8.0 if p_idx == 1 else 12.0)
            self.battery_soc = max(5.0, self.battery_soc - soc_drain)

            if p_idx == 1 and s_idx >= 1:
                # Sangat ideal: Surplus tipis dikombinasikan dengan baterai sedang/tinggi
                reward = +15.0
                status_note = "Optimal Eco-Balance: Menyeimbangkan surplus tipis dengan cadangan baterai."
            elif p_idx == 2 and t_idx >= 1:
                # Bagus: Memanfaatkan PV melimpah saat tarif normal/mahal tanpa menarik grid
                reward = +14.0
                status_note = "Green Arbitrage: Mengisi EV dengan energi surya saat tarif non-murah."
            elif p_idx == 0 and s_idx == 0:
                # Berbahaya: Memaksa eco charging saat daya defisit dan baterai sudah kritis
                reward = -12.0
                status_note = "Battery Depletion Risk: Baterai kritis dipaksa menyuplai beban defisit."
            else:
                reward = +7.0
                status_note = "Moderate Eco Charging: Pengisian ramah lingkungan berjalan stabil."

        elif action == 2:
            # AKSI 2: FAST / FULL CHARGING (PV + GRID) -> Beban 22 kW
            # Kecepatan pengisian maksimal, tetapi menarik daya besar dari jaringan utama
            soc_drain = 2.0 if p_idx == 2 else (6.0 if p_idx == 1 else 10.0)
            self.battery_soc = max(5.0, self.battery_soc - soc_drain)

            if p_idx == 2 and t_idx == 0:
                # Paling Optimal: Surplus PV melimpah + Tarif listrik murah!
                reward = +18.0
                status_note = "Peak Efficiency: Fast charging maksimal saat PV melimpah & tarif murah."
            elif p_idx == 2 and t_idx == 1 and s_idx >= 1:
                # Baik: Surplus PV melimpah dengan dukungan baterai memadai
                reward = +13.0
                status_note = "High Throughput: Pengisian cepat didukung surplus daya surya."
            elif p_idx == 0 or t_idx == 2:
                # Penalti Berat: Menyebabkan Grid Overload saat defisit atau lonjakan biaya saat tarif puncak!
                reward = -16.0
                status_note = "Grid Overload / Tariff Spike Penalty: Fast charging saat defisit atau tarif mahal!"
            else:
                reward = -3.0
                status_note = "Suboptimal Fast Charge: Menarik daya grid berlebih saat surplus tipis."

        # Majukan waktu simulasi 1 jam
        self.step_count += 1
        self.current_hour = (self.current_hour + 1) % 24
        done = self.step_count >= self.max_steps

        # Hitung state berikutnya (s')
        next_p, next_s, next_t, solar_kw, net_kw = self._simulate_microgrid_physics(self.current_hour)
        self.state_tuple = (next_p, next_s, next_t)
        next_state_id = self.encode_state(next_p, next_s, next_t)

        info = {
            "hour": self.current_hour,
            "solar_kw": round(solar_kw, 2),
            "net_power_kw": round(net_kw, 2),
            "battery_soc": round(self.battery_soc, 2),
            "status_note": status_note
        }
        return next_state_id, reward, done, info
