import numpy as np
import pandas as pd


class QLearningAgent:
    """
    Agen Reinforcement Learning Model-Free, Off-Policy berbasis Q-Table.
    """

    def __init__(
        self,
        n_states: int = 27,
        n_actions: int = 3,
        learning_rate: float = 0.8,
        discount_factor: float = 0.9,
        epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        decay_rate: float = 0.002,
        seed: int = 42
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate       # Learning rate (alpha)
        self.gamma = discount_factor     # Discount factor (gamma)
        self.epsilon = epsilon           # Probabilitas eksplorasi awal
        self.max_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.rng = np.random.default_rng(seed)

        # Inisialisasi Q-Table berukuran (27 x 3) dengan nilai 0.0
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    def choose_action(self, state: int, greedy_only: bool = False) -> int:
        """
        Memilih aksi berdasarkan strategi Epsilon-Greedy:
        - Dengan probabilitas epsilon: Eksplorasi (pilih aksi acak).
        - Dengan probabilitas 1 - epsilon: Eksploitasi (pilih argmax Q(s, a)).
        """
        if not greedy_only and self.rng.uniform(0.0, 1.0) < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        else:
            # Jika terdapat nilai Q maksimum yang seri, pilih secara acak di antaranya
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(np.isclose(q_values, max_q))[0]
            return int(self.rng.choice(best_actions))

    def update_q_value(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Memperbarui sel Q[state, action] menggunakan Persamaan Bellman.
        """
        old_q = self.q_table[state, action]
        max_future_q = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * max_future_q
        self.q_table[state, action] = (1.0 - self.alpha) * old_q + self.alpha * td_target

    def decay_epsilon(self, episode: int):
        """
        Menurunkan nilai epsilon secara eksponensial seiring bertambahnya episode
        agar agen beralih dari fase Eksplorasi ke fase Eksploitasi.
        """
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
            -self.decay_rate * episode
        )

    def train(self, env, total_episodes: int = 3000):
        """
        Menjalankan loop pelatihan Q-Learning selama `total_episodes`.
        Mengembalikan riwayat total reward dan epsilon pada setiap episode.
        """
        rewards_history = []
        epsilon_history = []

        for episode in range(total_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.choose_action(state, greedy_only=False)
                next_state, reward, done, _ = env.step(action)
                self.update_q_value(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            self.decay_epsilon(episode)
            rewards_history.append(total_reward)
            epsilon_history.append(self.epsilon)

        return rewards_history, epsilon_history

    def extract_policy(self, env) -> pd.DataFrame:
        """
        Mengekstrak Kebijakan Optimal (Optimal Policy) untuk seluruh 27 state
        dalam bentuk DataFrame terstruktur lengkap dengan nilai Q dan interpretasi.
        """
        rows = []
        for s in range(self.n_states):
            p_idx, s_idx, t_idx = env.decode_state(s)
            q_vals = self.q_table[s]
            best_action = int(np.argmax(q_vals))
            rows.append({
                "State_ID": s,
                "Kondisi_Daya": env.power_labels[p_idx],
                "Level_Baterai": env.soc_labels[s_idx],
                "Tarif_Listrik": env.tariff_labels[t_idx],
                "Q_Standby": round(float(q_vals[0]), 3),
                "Q_Eco_Charge": round(float(q_vals[1]), 3),
                "Q_Fast_Charge": round(float(q_vals[2]), 3),
                "Aksi_Optimal_ID": best_action,
                "Kebijakan_Optimal": env.action_labels[best_action]
            })
        return pd.DataFrame(rows)
