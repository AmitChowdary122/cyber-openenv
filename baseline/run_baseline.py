# baseline/run_baseline.py
import requests
import re

BASE_URL = "http://localhost:8000"

def extract_ip_from_logs(logs):
    """Extract IP address from logs looking for pattern 'from X.X.X.X'."""
    for log in logs:
        match = re.search(r'from (\d+\.\d+\.\d+\.\d+)', log)
        if match:
            return match.group(1)
    return None

def run_baseline():
    try:
        # Reset environment
        resp = requests.post(f"{BASE_URL}/reset")
        resp.raise_for_status()
        print("Reset successful")

        # Step 1: analyze_log
        action = {"action_type": "analyze_log", "parameters": {}}
        resp = requests.post(f"{BASE_URL}/step", json=action)
        resp.raise_for_status()
        data = resp.json()
        print(f"After analyze_log: reward={data['reward']}, threat={data['state']['system_state']['threat_level']}")
        logs = data['observation']['logs']

        # Extract attacker IP from logs
        attacker_ip = extract_ip_from_logs(logs)
        if attacker_ip:
            print(f"Extracted attacker IP: {attacker_ip}")

            # Step 2: identify_attacker
            action = {"action_type": "identify_attacker", "parameters": {"ip": attacker_ip}}
            resp = requests.post(f"{BASE_URL}/step", json=action)
            resp.raise_for_status()
            data = resp.json()
            print(f"After identify_attacker: reward={data['reward']}, threat={data['state']['system_state']['threat_level']}")

            # Step 3: block_ip
            action = {"action_type": "block_ip", "parameters": {"ip": attacker_ip}}
            resp = requests.post(f"{BASE_URL}/step", json=action)
            resp.raise_for_status()
            data = resp.json()
            print(f"After block_ip: reward={data['reward']}, threat={data['state']['system_state']['threat_level']}")

        else:
            print("No attacker IP found in logs, skipping identification and blocking.")

        # Step 4: quarantine_system
        action = {"action_type": "quarantine_system", "parameters": {}}
        resp = requests.post(f"{BASE_URL}/step", json=action)
        resp.raise_for_status()
        data = resp.json()
        print(f"After quarantine_system: reward={data['reward']}, threat={data['state']['system_state']['threat_level']}")

        # Get final grade
        resp = requests.get(f"{BASE_URL}/grade")
        resp.raise_for_status()
        score = resp.json()['score']
        print(f"Final grade: {score}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_baseline()