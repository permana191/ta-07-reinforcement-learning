import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.environment import SmartEVChargingEnv
from src.q_learning_agent import QLearningAgent

app = Flask(__name__)

Q_TABLE_PATH = "outputs/models/q_table_optimal.npy"
POLICY_CSV_PATH = "outputs/models/optimal_policy_table.csv"
METRICS_JSON_PATH = "outputs/models/training_metrics.json"


def ensure_trained_artifacts():
    """Memastikan Q-Table sudah tersedia sebelum server Flask menerima request."""
    if not (os.path.exists(Q_TABLE_PATH) and os.path.exists(METRICS_JSON_PATH)):
        import train_model
        train_model.main()


ensure_trained_artifacts()
env = SmartEVChargingEnv(max_steps=24, seed=42)
q_table = np.load(Q_TABLE_PATH)
policy_df = pd.read_csv(POLICY_CSV_PATH)
with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
    metrics_data = json.load(f)


@app.route("/")
def index():
    return render_template(
        "index.html",
        metrics=metrics_data,
        policy_rows=policy_df.to_dict(orient="records")
    )


@app.route("/api/predict_state", methods=["POST"])
def predict_state():
    """Endpoint interaktif untuk menguji keputusan Q-Table pada state tertentu."""
    data = request.get_json(force=True)
    power_idx = int(data.get("power_idx", 2))
    soc_idx = int(data.get("soc_idx", 1))
    tariff_idx = int(data.get("tariff_idx", 0))

    state_id = env.encode_state(power_idx, soc_idx, tariff_idx)
    q_vals = q_table[state_id].tolist()
    best_action = int(np.argmax(q_vals))

    explanations = {
        0: "Agen memilih STANDBY / TUNDA CHARGING untuk melindungi jaringan dari risiko overload atau lonjakan tarif puncak, sekaligus memberi waktu bagi panel surya mengisi ulang baterai stasiun.",
        1: "Agen memilih ECO CHARGING (PV + Baterai) karena kondisi daya berada pada surplus moderat atau tarif cukup tinggi, sehingga energi hijau dimanfaatkan optimal tanpa membebani jaringan utama.",
        2: "Agen memilih FAST / FULL CHARGING (PV + Grid) karena pasokan energi surya sedang melimpah dan tarif listrik berada pada kategori ekonomis untuk memaksimalkan layanan EV."
    }

    return jsonify({
        "state_id": state_id,
        "state_desc": env.get_state_description(state_id),
        "q_values": {
            "standby": round(q_vals[0], 3),
            "eco": round(q_vals[1], 3),
            "fast": round(q_vals[2], 3)
        },
        "best_action_id": best_action,
        "best_action_label": env.action_labels[best_action],
        "explanation": explanations[best_action]
    })


@app.route("/api/simulate_24h", methods=["POST"])
def simulate_24h():
    """Menjalankan simulasi 1 episode penuh (24 jam) menggunakan kebijakan Q-Table terlatih."""
    sim_env = SmartEVChargingEnv(max_steps=24)
    agent = QLearningAgent(n_states=27, n_actions=3)
    agent.q_table = q_table.copy()

    state, info = sim_env.reset()
    timeline = []
    cumulative_reward = 0.0

    for step in range(24):
        hour_now = info["hour"]
        soc_before = info["battery_soc"]
        state_desc = sim_env.get_state_description(state)
        action = agent.choose_action(state, greedy_only=True)
        next_state, reward, done, info = sim_env.step(action)
        cumulative_reward += reward

        timeline.append({
            "step": step + 1,
            "hour": f"{hour_now:02d}:00",
            "state_id": state,
            "state_desc": state_desc,
            "action_id": action,
            "action_label": sim_env.action_labels[action],
            "reward": round(reward, 1),
            "cumulative_reward": round(cumulative_reward, 1),
            "battery_soc": info["battery_soc"],
            "note": info["status_note"]
        })
        state = next_state

    return jsonify({
        "total_reward": round(cumulative_reward, 1),
        "timeline": timeline
    })


@app.route("/plot/<path:filename>")
def serve_plot(filename):
    from flask import send_from_directory
    return send_from_directory("outputs/plots", filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
