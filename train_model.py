import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.environment import SmartEVChargingEnv
from src.q_learning_agent import QLearningAgent

# Konfigurasi gaya grafik akademik
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def evaluate_agent(env, agent, n_eval_episodes: int = 200) -> dict:
    """
    Mengevaluasi performa kebijakan optimal (greedy_only=True) setelah pelatihan selesai.
    """
    eval_rewards = []
    overload_count = 0
    total_steps = 0

    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.choose_action(state, greedy_only=True)
            next_state, reward, done, _ = env.step(action)
            if reward <= -15.0:
                overload_count += 1
            total_steps += 1
            ep_reward += reward
            state = next_state
        eval_rewards.append(ep_reward)

    return {
        "mean_reward": round(float(np.mean(eval_rewards)), 2),
        "std_reward": round(float(np.std(eval_rewards)), 2),
        "max_reward": round(float(np.max(eval_rewards)), 2),
        "min_reward": round(float(np.min(eval_rewards)), 2),
        "overload_rate_pct": round((overload_count / total_steps) * 100.0, 2)
    }


def main():
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    print("=" * 72)
    print("TA-06 REINFORCEMENT LEARNING: SMART EV CHARGING SCHEDULER (Q-LEARNING)")
    print("=" * 72)

    # --------------------------------------------------------------------------
    # LANGKAH 1 & 2: PERSIAPAN LINGKUNGAN & PELATIHAN MODEL UTAMA (DEFAULT)
    # --------------------------------------------------------------------------
    TOTAL_EPISODES = 3000
    WINDOW_SIZE = 100

    env_main = SmartEVChargingEnv(max_steps=24, seed=42)
    agent_main = QLearningAgent(
        n_states=env_main.n_states,
        n_actions=env_main.n_actions,
        learning_rate=0.8,    # Alpha default sesuai Modul 07
        discount_factor=0.9,  # Gamma default sesuai Modul 07
        epsilon=1.0,
        min_epsilon=0.01,
        decay_rate=0.002,
        seed=42
    )

    print(f"[1/5] Melatih Agen Q-Learning Utama ({TOTAL_EPISODES} episode, 24 jam/episode)...")
    rewards_main, epsilons_main = agent_main.train(env_main, total_episodes=TOTAL_EPISODES)

    # Simpan Q-Table dan Tabel Kebijakan Optimal
    np.save("outputs/models/q_table_optimal.npy", agent_main.q_table)
    policy_df = agent_main.extract_policy(env_main)
    policy_df.to_csv("outputs/models/optimal_policy_table.csv", index=False)

    # Evaluasi performa akhir (Eksploitasi murni)
    eval_env = SmartEVChargingEnv(max_steps=24, seed=100)
    eval_stats = evaluate_agent(eval_env, agent_main, n_eval_episodes=200)
    print(f"      -> Rata-rata Reward Evaluasi (200 Episode): {eval_stats['mean_reward']} +/- {eval_stats['std_reward']}")
    print(f"      -> Tingkat Risiko Grid Overload          : {eval_stats['overload_rate_pct']}%")

    # --------------------------------------------------------------------------
    # LANGKAH 3: VISUALISASI KURVA PEMBELAJARAN (MOVING AVERAGE & EPSILON DECAY)
    # --------------------------------------------------------------------------
    print("[2/5] Membuat Plot 1: Kurva Pembelajaran & Peluruhan Epsilon...")
    rewards_ma = pd.Series(rewards_main).rolling(window=WINDOW_SIZE, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=150)
    ax1.plot(rewards_main, color="#93C5FD", alpha=0.35, linewidth=0.8, label="Reward per Episode (Mentah)")
    ax1.plot(rewards_ma, color="#1D4ED8", linewidth=2.4, label=f"Moving Average ({WINDOW_SIZE} Episode)")
    ax1.set_title("Kurva Pembelajaran Q-Learning: Akumulasi Reward & Peluruhan Epsilon", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Episode Pelatihan", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Reward per Episode (24 Jam)", fontsize=11, fontweight="bold", color="#1D4ED8")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(epsilons_main, color="#DC2626", linestyle="--", linewidth=1.8, label="Nilai Epsilon (Eksplorasi)")
    ax2.set_ylabel("Probabilitas Eksplorasi (Epsilon)", fontsize=11, fontweight="bold", color="#DC2626")
    ax2.set_ylim(0, 1.08)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True, facecolor="white")

    plt.tight_layout()
    plt.savefig("outputs/plots/01_learning_curve.png")
    plt.close()

    # --------------------------------------------------------------------------
    # LANGKAH 4: VISUALISASI KEBIJAKAN OPTIMAL (OPTIMAL POLICY GRID & Q-HEATMAP)
    # --------------------------------------------------------------------------
    print("[3/5] Membuat Plot 2 & 3: Peta Kebijakan Optimal (27 States) & Heatmap Q-Table...")
    # Susun 27 states ke dalam grid 9 baris (Kondisi Daya x SOC) x 3 kolom (Tarif Listrik)
    policy_matrix = np.zeros((9, 3), dtype=int)
    annot_matrix = np.empty((9, 3), dtype=object)
    row_labels = []
    short_actions = ["STANDBY (0)", "ECO CHARGE (1)", "FAST CHARGE (2)"]

    for p in range(3):
        for s in range(3):
            row_idx = p * 3 + s
            row_labels.append(f"{env_main.power_labels[p]} | {env_main.soc_labels[s]}")
            for t in range(3):
                state_id = env_main.encode_state(p, s, t)
                best_act = int(np.argmax(agent_main.q_table[state_id]))
                policy_matrix[row_idx, t] = best_act
                annot_matrix[row_idx, t] = f"S{state_id}: {short_actions[best_act]}"

    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=150)
    cmap_policy = sns.color_palette(["#F59E0B", "#10B981", "#2563EB"])
    sns.heatmap(
        policy_matrix,
        annot=annot_matrix,
        fmt="",
        cmap=cmap_policy,
        cbar=False,
        linewidths=1.5,
        linecolor="white",
        xticklabels=["Tarif Murah (Off-Peak)", "Tarif Normal (Mid-Peak)", "Tarif Mahal (Peak Load)"],
        yticklabels=row_labels,
        ax=ax,
        annot_kws={"size": 9.5, "weight": "bold", "color": "white"}
    )
    ax.set_title("Peta Kebijakan Optimal (Optimal Policy) pada 27 State Microgrid PV-EV", fontsize=12.5, fontweight="bold", pad=14)
    ax.set_xlabel("Kategori Tarif Listrik Dinamis (Time-of-Use)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Kombinasi Kondisi Daya Bersih & Level Baterai (SOC)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/plots/02_optimal_policy_map.png")
    plt.close()

    # Plot 3: Heatmap Nilai Q(s, a) lengkap untuk seluruh 27 state
    fig, ax = plt.subplots(figsize=(9, 10), dpi=150)
    state_y_labels = [f"S{s:02d}: {env_main.get_state_description(s)}" for s in range(27)]
    sns.heatmap(
        agent_main.q_table,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.6,
        xticklabels=["A0: Standby", "A1: Eco Charging", "A2: Fast Charging"],
        yticklabels=state_y_labels,
        ax=ax,
        annot_kws={"size": 8.5}
    )
    ax.set_title("Distribusi Nilai Q-Table Akhir Q(s, a) untuk 27 State & 3 Aksi", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Ruang Aksi (Action Space)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Ruang Keadaan Diskrit (State Space)", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/plots/03_q_values_heatmap.png")
    plt.close()

    # --------------------------------------------------------------------------
    # LANGKAH 5: EKSPERIMEN HIPERPARAMETER (GAMMA TINGGI VS RENDAH & ALPHA)
    # --------------------------------------------------------------------------
    print("[4/5] Menjalankan Eksperimen Hiperparameter (Gamma & Alpha)...")
    # Eksperimen A: Discount Factor Gamma = 0.90 (Visioner) vs Gamma = 0.10 (Miopik)
    env_g_low = SmartEVChargingEnv(max_steps=24, seed=42)
    agent_g_low = QLearningAgent(learning_rate=0.8, discount_factor=0.1, decay_rate=0.002, seed=42)
    rewards_g_low, _ = agent_g_low.train(env_g_low, total_episodes=TOTAL_EPISODES)

    ma_g_high = pd.Series(rewards_main).rolling(window=WINDOW_SIZE, min_periods=1).mean()
    ma_g_low = pd.Series(rewards_g_low).rolling(window=WINDOW_SIZE, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10.5, 5), dpi=150)
    ax.plot(ma_g_high, color="#1D4ED8", linewidth=2.3, label="Gamma = 0.90 (Visioner / Jangka Panjang)")
    ax.plot(ma_g_low, color="#EA580C", linewidth=2.3, label="Gamma = 0.10 (Miopik / Reward Instan)")
    ax.set_title("Eksperimen Hiperparameter Discount Factor (γ): Visioner vs Miopik", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Episode Pelatihan", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Moving Average Reward (Window {WINDOW_SIZE})", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs/plots/04_hyperparam_gamma.png")
    plt.close()

    # Eksperimen B: Learning Rate Alpha = 0.80 (Cepat) vs Alpha = 0.05 (Sangat Lambat)
    env_a_low = SmartEVChargingEnv(max_steps=24, seed=42)
    agent_a_low = QLearningAgent(learning_rate=0.05, discount_factor=0.9, decay_rate=0.002, seed=42)
    rewards_a_low, _ = agent_a_low.train(env_a_low, total_episodes=TOTAL_EPISODES)

    ma_a_high = pd.Series(rewards_main).rolling(window=WINDOW_SIZE, min_periods=1).mean()
    ma_a_low = pd.Series(rewards_a_low).rolling(window=WINDOW_SIZE, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10.5, 5), dpi=150)
    ax.plot(ma_a_high, color="#059669", linewidth=2.3, label="Alpha = 0.80 (Pembaruan Q-Table Agresif/Cepat)")
    ax.plot(ma_a_low, color="#7C3AED", linewidth=2.3, label="Alpha = 0.05 (Pembaruan Q-Table Lambat/Konservatif)")
    ax.set_title("Eksperimen Hiperparameter Learning Rate (α): Cepat (0.80) vs Lambat (0.05)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Episode Pelatihan", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Moving Average Reward (Window {WINDOW_SIZE})", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs/plots/05_hyperparam_alpha.png")
    plt.close()

    # Simpan ringkasan metrik ke JSON untuk dikonsumsi oleh Dashboard Web (app.py)
    summary_metrics = {
        "scenario_title": "Smart EV Charging Scheduler pada Microgrid PV-EV",
        "student_name": "Sigit Miraj Permana",
        "student_nim": "301240037",
        "total_states": 27,
        "total_actions": 3,
        "total_episodes": TOTAL_EPISODES,
        "default_hyperparams": {"alpha": 0.8, "gamma": 0.9, "decay_rate": 0.002},
        "evaluation": eval_stats,
        "gamma_comparison": {
            "gamma_0_90_final_ma": round(float(ma_g_high.iloc[-1]), 2),
            "gamma_0_10_final_ma": round(float(ma_g_low.iloc[-1]), 2)
        },
        "alpha_comparison": {
            "alpha_0_80_final_ma": round(float(ma_a_high.iloc[-1]), 2),
            "alpha_0_05_final_ma": round(float(ma_a_low.iloc[-1]), 2)
        },
        "action_distribution": policy_df["Kebijakan_Optimal"].value_counts().to_dict()
    }
    with open("outputs/models/training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)

    print("[5/5] Seluruh pelatihan, visualisasi, dan penyimpanan model selesai!")
    print("=" * 72)


if __name__ == "__main__":
    main()
