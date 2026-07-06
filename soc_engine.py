"""
AICS-106 — Live-Style AI SOC Threat Detection Engine
======================================================
Loads the trained deep residual NN (network flow classifier) and LSTM
(auth session anomaly classifier), simulates a live stream of security
telemetry, classifies each event, assigns a severity level, maps
detections to MITRE ATT&CK-style tactics/techniques, and writes
analyst-ready JSON and Markdown incident reports.

This is a defensive detection tool only. It does not execute, generate,
or deploy any offensive capability. All input data is synthetic.
"""

import json
import time
import random
import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RANDOM_SEED = 7
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

N_EVENTS_TO_STREAM = 15          # how many events to simulate in this run
STREAM_DELAY_SECONDS = 0.6       # pause between events, for a "live" feel
INCIDENT_OUTPUT_DIR = "incident_reports"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# MODEL ARCHITECTURE DEFINITIONS (must match training notebooks exactly)
# ----------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResidualNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_blocks=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.classifier(x)


class AuthLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, event_seq, time_seq):
        embedded = self.embedding(event_seq)
        time_feat = time_seq.unsqueeze(-1)
        lstm_input = torch.cat([embedded, time_feat], dim=-1)
        _, (h_n, _) = self.lstm(lstm_input)
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.classifier(final_hidden)


# ----------------------------------------------------------------------
# ATT&CK-STYLE MAPPING (illustrative, defensive use only)
# ----------------------------------------------------------------------
ATTACK_MAPPING = {
    "DoS": {"tactic": "Impact", "technique": "T1499 - Endpoint Denial of Service"},
    "PortScan": {"tactic": "Discovery", "technique": "T1046 - Network Service Discovery"},
    "BruteForce": {"tactic": "Credential Access", "technique": "T1110 - Brute Force"},
    "Botnet": {"tactic": "Command and Control", "technique": "T1071 - Application Layer Protocol"},
    "WebAttack": {"tactic": "Initial Access", "technique": "T1190 - Exploit Public-Facing Application"},
    "Benign": {"tactic": "N/A", "technique": "N/A"},
    "AuthAnomaly_BruteForce": {"tactic": "Credential Access", "technique": "T1110 - Brute Force (Credential Stuffing Pattern)"},
    "AuthAnomaly_PrivEsc": {"tactic": "Privilege Escalation", "technique": "T1548 - Abuse Elevation Control Mechanism"},
}

SEVERITY_ORDER = ["Info", "Low", "Medium", "High", "Critical"]

FLOW_SEVERITY = {
    "Benign": "Info",
    "PortScan": "Low",
    "BruteForce": "Medium",
    "WebAttack": "High",
    "Botnet": "High",
    "DoS": "Critical",
}


def confidence_bump(base_severity, confidence):
    """Bump severity up one level if the model is very confident (>0.9)."""
    if confidence > 0.9 and base_severity in SEVERITY_ORDER:
        idx = SEVERITY_ORDER.index(base_severity)
        return SEVERITY_ORDER[min(idx + 1, len(SEVERITY_ORDER) - 1)]
    return base_severity


# ----------------------------------------------------------------------
# LOAD TRAINED MODELS
# ----------------------------------------------------------------------
def load_residual_nn():
    ckpt = torch.load("residual_nn_model.pt", map_location=device, weights_only=False)
    model = ResidualNN(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_classes=ckpt["num_classes"],
        num_blocks=ckpt["num_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = joblib.load("residual_nn_scaler.pkl")
    label_encoder = joblib.load("residual_nn_label_encoder.pkl")
    return model, scaler, label_encoder


def load_lstm():
    ckpt = torch.load("lstm_model.pt", map_location=device, weights_only=False)
    model = AuthLSTM(
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        num_classes=ckpt["num_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["event_to_idx"], ckpt["max_seq_len"]


# ----------------------------------------------------------------------
# PREDICTION HELPERS
# ----------------------------------------------------------------------
def predict_flow(model, scaler, label_encoder, flow_row):
    """flow_row: dict of feature_name -> value (one network flow, no label)"""
    protocols = ["TCP", "UDP", "ICMP"]
    proto_onehot = {f"proto_{p}": 1.0 if flow_row["protocol"] == p else 0.0 for p in protocols}

    feature_order = [
        "duration", "src_port", "dst_port", "src_bytes", "dst_bytes",
        "packet_count", "syn_count", "fin_count", "avg_packet_size",
        "flow_rate", "flow_iat_mean", "flow_iat_std",
    ] + [f"proto_{p}" for p in protocols]

    row = {**flow_row, **proto_onehot}
    x = np.array([[row[f] for f in feature_order]], dtype=np.float32)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])
    return pred_label, confidence


def predict_session(model, event_to_idx, max_seq_len, session_events):
    """session_events: list of (event_type, time_gap_seconds) tuples, ordered."""
    events = [event_to_idx.get(evt, 0) for evt, _ in session_events][:max_seq_len]
    times = [np.log1p(max(gap, 0)) / 10.0 for _, gap in session_events][:max_seq_len]

    padded_events = np.zeros(max_seq_len, dtype=np.int64)
    padded_times = np.zeros(max_seq_len, dtype=np.float32)
    padded_events[:len(events)] = events
    padded_times[:len(times)] = times

    xe = torch.tensor(np.array([padded_events]), dtype=torch.long).to(device)
    xt = torch.tensor(np.array([padded_times]), dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(xe, xt)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    label = "Anomalous" if pred_idx == 1 else "Normal"
    confidence = float(probs[pred_idx])
    return label, confidence


# ----------------------------------------------------------------------
# INCIDENT REPORT GENERATION
# ----------------------------------------------------------------------
def write_incident_report(incident):
    os.makedirs(INCIDENT_OUTPUT_DIR, exist_ok=True)
    incident_id = incident["incident_id"]

    json_path = os.path.join(INCIDENT_OUTPUT_DIR, f"{incident_id}.json")
    with open(json_path, "w") as f:
        json.dump(incident, f, indent=2)

    md_path = os.path.join(INCIDENT_OUTPUT_DIR, f"{incident_id}.md")
    with open(md_path, "w") as f:
        f.write(f"# Incident Report: {incident_id}\n\n")
        f.write(f"**Timestamp:** {incident['timestamp']}\n\n")
        f.write(f"**Source Type:** {incident['source_type']}\n\n")
        f.write(f"**Detected Threat:** {incident['predicted_label']}\n\n")
        f.write(f"**Confidence:** {incident['confidence']:.1%}\n\n")
        f.write(f"**Severity:** {incident['severity']}\n\n")
        f.write(f"## ATT&CK Mapping\n")
        f.write(f"- **Tactic:** {incident['attack_tactic']}\n")
        f.write(f"- **Technique:** {incident['attack_technique']}\n\n")
        f.write(f"## Event Details\n")
        for k, v in incident["event_summary"].items():
            f.write(f"- **{k}:** {v}\n")
        f.write(f"\n## Recommended Action\n{incident['recommended_action']}\n")

    return json_path, md_path


def recommended_action(label, severity):
    actions = {
        "DoS": "Rate-limit or block source IP at the firewall/load balancer; verify service availability.",
        "Botnet": "Isolate affected host from the network; inspect for C2 beaconing; run endpoint scan.",
        "BruteForce": "Lock affected account(s); enforce MFA; review source IP reputation.",
        "WebAttack": "Review web server logs for the target endpoint; apply WAF rule if applicable; check for successful exploitation.",
        "PortScan": "Monitor source IP for follow-up activity; consider blocking if scanning continues.",
        "Anomalous": "Review session activity with the account owner; verify login was authorized; consider forced password reset if suspicious.",
    }
    return actions.get(label, "Review event manually; no automated recommendation available.")


# ----------------------------------------------------------------------
# SIMULATED LIVE STREAM
# ----------------------------------------------------------------------
def simulate_stream():
    print("=" * 78)
    print("AICS-106 — AI SOC Threat Detection Engine — LIVE MODE")
    print("=" * 78)
    print(f"Loading models...")

    residual_model, scaler, label_encoder = load_residual_nn()
    lstm_model, event_to_idx, max_seq_len = load_lstm()

    print("Models loaded successfully.\n")

    flows_df = pd.read_csv("network_flows.csv")
    auth_df = pd.read_csv("auth_logs.csv", parse_dates=["timestamp"])
    auth_df = auth_df.sort_values(["session_id", "timestamp"])
    session_groups = list(auth_df.groupby("session_id"))

    incident_count = 0

    for i in range(N_EVENTS_TO_STREAM):
        event_source = random.choice(["network_flow", "auth_session"])
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        if event_source == "network_flow":
            row = flows_df.sample(1).iloc[0]
            flow_dict = row.drop("label").to_dict()
            true_label = row["label"]  # for display only; not used by the model

            pred_label, confidence = predict_flow(residual_model, scaler, label_encoder, flow_dict)

            print(f"[{i+1:02d}/{N_EVENTS_TO_STREAM}] NETWORK FLOW  | "
                  f"src_port={int(flow_dict['src_port'])} dst_port={int(flow_dict['dst_port'])} "
                  f"proto={flow_dict['protocol']}  ->  Predicted: {pred_label} ({confidence:.1%})")

            if pred_label != "Benign":
                incident_count += 1
                severity = confidence_bump(FLOW_SEVERITY.get(pred_label, "Medium"), confidence)
                mapping = ATTACK_MAPPING.get(pred_label, {"tactic": "Unknown", "technique": "Unknown"})

                incident = {
                    "incident_id": f"INC-{timestamp.replace(':', '').replace('-', '')}-{i+1:03d}",
                    "timestamp": timestamp,
                    "source_type": "network_flow",
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "severity": severity,
                    "attack_tactic": mapping["tactic"],
                    "attack_technique": mapping["technique"],
                    "event_summary": {
                        "protocol": flow_dict["protocol"],
                        "src_port": int(flow_dict["src_port"]),
                        "dst_port": int(flow_dict["dst_port"]),
                        "packet_count": int(flow_dict["packet_count"]),
                        "duration_sec": round(float(flow_dict["duration"]), 4),
                    },
                    "recommended_action": recommended_action(pred_label, severity),
                }
                json_path, md_path = write_incident_report(incident)
                print(f"        >>> INCIDENT RAISED [{severity}] — {mapping['tactic']} / {mapping['technique']}")
                print(f"        >>> Report written: {json_path}")

        else:
            session_id, group = random.choice(session_groups)
            events = list(zip(
                group["event_type"].tolist(),
                group["timestamp"].diff().dt.total_seconds().fillna(0).tolist(),
            ))

            pred_label, confidence = predict_session(lstm_model, event_to_idx, max_seq_len, events)

            print(f"[{i+1:02d}/{N_EVENTS_TO_STREAM}] AUTH SESSION  | "
                  f"session_id={session_id} user={group['user'].iloc[0]} events={len(events)}  ->  "
                  f"Predicted: {pred_label} ({confidence:.1%})")

            if pred_label == "Anomalous":
                incident_count += 1
                severity = confidence_bump("High", confidence)

                # Heuristic sub-classification for ATT&CK mapping based on event pattern
                sudo_count = sum(1 for e, _ in events if e == "sudo_success")
                fail_count = sum(1 for e, _ in events if e == "login_failed")

                if fail_count >= 3:
                    mapping_key = "AuthAnomaly_BruteForce"
                elif sudo_count >= 2:
                    mapping_key = "AuthAnomaly_PrivEsc"
                else:
                    mapping_key = "AuthAnomaly_BruteForce"

                mapping = ATTACK_MAPPING[mapping_key]

                incident = {
                    "incident_id": f"INC-{timestamp.replace(':', '').replace('-', '')}-{i+1:03d}",
                    "timestamp": timestamp,
                    "source_type": "auth_session",
                    "predicted_label": "Anomalous",
                    "confidence": confidence,
                    "severity": severity,
                    "attack_tactic": mapping["tactic"],
                    "attack_technique": mapping["technique"],
                    "event_summary": {
                        "session_id": int(session_id),
                        "user": group["user"].iloc[0],
                        "host": group["host"].iloc[0],
                        "num_events": len(events),
                        "failed_logins": fail_count,
                        "sudo_events": sudo_count,
                    },
                    "recommended_action": recommended_action("Anomalous", severity),
                }
                json_path, md_path = write_incident_report(incident)
                print(f"        >>> INCIDENT RAISED [{severity}] — {mapping['tactic']} / {mapping['technique']}")
                print(f"        >>> Report written: {json_path}")

        time.sleep(STREAM_DELAY_SECONDS)

    print("\n" + "=" * 78)
    print(f"Stream complete. {incident_count} incident(s) raised out of {N_EVENTS_TO_STREAM} events processed.")
    print(f"Incident reports saved in ./{INCIDENT_OUTPUT_DIR}/")
    print("=" * 78)


if __name__ == "__main__":
    simulate_stream()
