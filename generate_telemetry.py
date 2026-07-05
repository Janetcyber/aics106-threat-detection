"""
AICS-106 — Synthetic Enterprise Telemetry Generator
=====================================================
Generates two datasets for the Deep Learning Threat Detection lab:

1. network_flows.csv   -> for the deep residual NN (multiclass flow classifier)
2. auth_logs.csv       -> for the sequence model (Linux auth log anomaly detection)

This is 100% synthetic, offline, and defensive in nature — no real traffic,
no exploit code, no credential data. Safe under the lab's ethics rules.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_FLOW_ROWS = 100_000
FLOW_CLASSES = ["Benign", "DoS", "PortScan", "BruteForce", "Botnet", "WebAttack"]

FLOW_CLASS_WEIGHTS = {
    "Benign": 0.55,
    "DoS": 0.15,
    "PortScan": 0.12,
    "BruteForce": 0.08,
    "Botnet": 0.06,
    "WebAttack": 0.04,
}

N_AUTH_SESSIONS = 8_000
MAX_EVENTS_PER_SESSION = 12
USERS = [f"user{i:03d}" for i in range(1, 61)] + ["root", "admin", "svc_backup", "svc_deploy"]
HOSTS = [f"host-{i:02d}.internal" for i in range(1, 21)]


def generate_network_flows(n_rows=N_FLOW_ROWS):
    labels = np.random.choice(
        list(FLOW_CLASS_WEIGHTS.keys()),
        size=n_rows,
        p=list(FLOW_CLASS_WEIGHTS.values()),
    )

    rows = []
    for label in labels:
        if label == "Benign":
            duration = np.random.exponential(2.0)
            src_bytes = np.random.lognormal(6, 1.2)
            dst_bytes = np.random.lognormal(6, 1.2)
            packet_count = np.random.poisson(20)
            syn_count = np.random.poisson(1)
            fin_count = np.random.poisson(1)
            dst_port = np.random.choice([80, 443, 22, 53, 3306, 8080])
            iat_mean = np.random.exponential(0.5)
            iat_std = np.random.exponential(0.2)

        elif label == "DoS":
            duration = np.random.exponential(0.3)
            src_bytes = np.random.lognormal(3, 0.5)
            dst_bytes = np.random.lognormal(1, 0.3)
            packet_count = np.random.poisson(500)
            syn_count = np.random.poisson(300)
            fin_count = np.random.poisson(0.5)
            dst_port = np.random.choice([80, 443])
            iat_mean = np.random.exponential(0.01)
            iat_std = np.random.exponential(0.005)

        elif label == "PortScan":
            duration = np.random.exponential(0.1)
            src_bytes = np.random.lognormal(2, 0.4)
            dst_bytes = np.random.lognormal(0.5, 0.3)
            packet_count = np.random.poisson(3)
            syn_count = np.random.poisson(2)
            fin_count = np.random.poisson(0.1)
            dst_port = np.random.randint(1, 65535)
            iat_mean = np.random.exponential(0.02)
            iat_std = np.random.exponential(0.01)

        elif label == "BruteForce":
            duration = np.random.exponential(1.0)
            src_bytes = np.random.lognormal(4, 0.6)
            dst_bytes = np.random.lognormal(3, 0.5)
            packet_count = np.random.poisson(15)
            syn_count = np.random.poisson(5)
            fin_count = np.random.poisson(4)
            dst_port = np.random.choice([22, 3389, 21, 23])
            iat_mean = np.random.exponential(0.3)
            iat_std = np.random.exponential(0.1)

        elif label == "Botnet":
            duration = np.random.exponential(5.0)
            src_bytes = np.random.lognormal(5, 0.8)
            dst_bytes = np.random.lognormal(5, 0.8)
            packet_count = np.random.poisson(40)
            syn_count = np.random.poisson(3)
            fin_count = np.random.poisson(3)
            dst_port = np.random.choice([6667, 6697, 4444, 8080, 443])
            iat_mean = np.random.exponential(1.5)
            iat_std = np.random.exponential(0.8)

        else:
            duration = np.random.exponential(0.8)
            src_bytes = np.random.lognormal(7, 1.0)
            dst_bytes = np.random.lognormal(4, 0.8)
            packet_count = np.random.poisson(25)
            syn_count = np.random.poisson(2)
            fin_count = np.random.poisson(2)
            dst_port = np.random.choice([80, 443, 8443])
            iat_mean = np.random.exponential(0.4)
            iat_std = np.random.exponential(0.2)

        protocol = np.random.choice(["TCP", "UDP", "ICMP"], p=[0.75, 0.20, 0.05])
        src_port = np.random.randint(1024, 65535)
        avg_packet_size = (src_bytes + dst_bytes) / max(packet_count, 1)
        flow_rate = (src_bytes + dst_bytes) / max(duration, 0.001)

        rows.append([
            round(duration, 4), protocol, src_port, int(dst_port),
            round(src_bytes, 2), round(dst_bytes, 2), int(packet_count),
            int(syn_count), int(fin_count), round(avg_packet_size, 2),
            round(flow_rate, 2), round(iat_mean, 4), round(iat_std, 4),
            label,
        ])

    cols = [
        "duration", "protocol", "src_port", "dst_port",
        "src_bytes", "dst_bytes", "packet_count",
        "syn_count", "fin_count", "avg_packet_size",
        "flow_rate", "flow_iat_mean", "flow_iat_std",
        "label",
    ]
    df = pd.DataFrame(rows, columns=cols)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def generate_auth_sessions(n_sessions=N_AUTH_SESSIONS):
    records = []
    session_id = 0
    base_time = datetime(2026, 6, 1, 0, 0, 0)

    for _ in range(n_sessions):
        session_id += 1
        user = random.choice(USERS)
        host = random.choice(HOSTS)
        is_anomalous = np.random.rand() < 0.12

        start_offset = np.random.randint(0, 60 * 24 * 30)
        t = base_time + timedelta(minutes=int(start_offset))

        events = []
        if not is_anomalous:
            t = t.replace(hour=np.random.randint(7, 19))
            n_events = np.random.randint(2, 6)
            events.append(("login_success", t))
            for i in range(n_events - 1):
                t += timedelta(seconds=np.random.randint(5, 600))
                evt = np.random.choice(
                    ["command_exec", "sudo_success", "logout"],
                    p=[0.7, 0.2, 0.1],
                )
                events.append((evt, t))
        else:
            pattern = np.random.choice(["brute_then_success", "off_hours_sudo", "rapid_sudo_abuse"])
            if pattern == "brute_then_success":
                n_fail = np.random.randint(4, MAX_EVENTS_PER_SESSION - 1)
                for i in range(n_fail):
                    t += timedelta(seconds=np.random.randint(1, 5))
                    events.append(("login_failed", t))
                t += timedelta(seconds=np.random.randint(1, 5))
                events.append(("login_success", t))
            elif pattern == "off_hours_sudo":
                t = t.replace(hour=np.random.choice([1, 2, 3, 4]))
                events.append(("login_success", t))
                for i in range(np.random.randint(2, 5)):
                    t += timedelta(seconds=np.random.randint(5, 60))
                    events.append(("sudo_success", t))
            else:
                events.append(("login_success", t))
                for i in range(np.random.randint(6, MAX_EVENTS_PER_SESSION)):
                    t += timedelta(seconds=np.random.randint(1, 3))
                    events.append(("sudo_success", t))

        for evt_type, ts in events:
            records.append([session_id, user, host, evt_type, ts, int(is_anomalous)])

    df = pd.DataFrame(records, columns=["session_id", "user", "host", "event_type", "timestamp", "session_label"])
    return df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)


if __name__ == "__main__":
    print("Generating network flow dataset...")
    flows = generate_network_flows()
    flows.to_csv("network_flows.csv", index=False)
    print(f"  -> network_flows.csv  ({len(flows):,} rows)")
    print(flows["label"].value_counts())

    print("\nGenerating auth log sequence dataset...")
    auth = generate_auth_sessions()
    auth.to_csv("auth_logs.csv", index=False)
    print(f"  -> auth_logs.csv  ({len(auth):,} events, {auth['session_id'].nunique():,} sessions)")
    print(auth.groupby("session_id")["session_label"].first().value_counts())

    print("\nDone. Both files saved in the current directory.")
