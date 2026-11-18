async function evaluateState() {
    const payload = {
        power_idx: document.getElementById("power_idx").value,
        soc_idx: document.getElementById("soc_idx").value,
        tariff_idx: document.getElementById("tariff_idx").value
    };
    const res = await fetch("/api/predict_state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    const data = await res.json();
    document.getElementById("resStateTag").textContent = `State S${String(data.state_id).padStart(2, "0")}`;
    document.getElementById("resActionLabel").textContent = data.best_action_label;
    document.getElementById("qVal0").textContent = data.q_values.standby;
    document.getElementById("qVal1").textContent = data.q_values.eco;
    document.getElementById("qVal2").textContent = data.q_values.fast;
    document.getElementById("resExplanation").textContent = data.explanation;
}

async function run24HSimulation() {
    const res = await fetch("/api/simulate_24h", { method: "POST" });
    const data = await res.json();
    const banner = document.getElementById("sim24Summary");
    banner.style.display = "block";
    banner.textContent = `Simulasi 24 Jam Selesai — Total Akumulasi Reward Episode: ${data.total_reward >= 0 ? "+" : ""}${data.total_reward} Poin`;

    const tbody = document.querySelector("#sim24Table tbody");
    tbody.innerHTML = "";
    data.timeline.forEach(item => {
        const tr = document.createElement("tr");
        const rewSign = item.reward >= 0 ? `+${item.reward}` : `${item.reward}`;
        tr.innerHTML = `
            <td><strong>${item.hour}</strong></td>
            <td>S${String(item.state_id).padStart(2, "0")}</td>
            <td>${item.state_desc}</td>
            <td><span class="policy-badge act-${item.action_id}">${item.action_label}</span></td>
            <td>${item.battery_soc}%</td>
            <td><strong>${rewSign}</strong></td>
            <td>${item.note}</td>
        `;
        tbody.appendChild(tr);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    evaluateState();
});
