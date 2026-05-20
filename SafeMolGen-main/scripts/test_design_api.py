#!/usr/bin/env python3
"""Quick test: call design API like a user (optional seed, first_iteration_temperature, etc.)."""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000/api/v1"

def main():
    # 1) Config
    with urllib.request.urlopen(f"{BASE}/config") as r:
        config = json.load(r)
    print("Config keys:", list(config.keys()))
    print("  first_iteration_temperature_default:", config.get("first_iteration_temperature_default"))
    print("  generator_early_available:", config.get("generator_early_available"))

    # 2) Design with scientist params: no seed, first_iteration_temperature 1.15, 1 iteration
    body = json.dumps({
        "target_success": 0.5,
        "max_iterations": 1,
        "first_iteration_temperature": 1.4,
        "selection_mode": "phase_weighted",
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/design",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        print("Design failed:", e.code, e.read().decode())
        sys.exit(1)
    except Exception as e:
        print("Design error:", e)
        sys.exit(1)

    print("\nDesign result:")
    print("  final_smiles:", out.get("final_smiles", "")[:60] + "..." if len(out.get("final_smiles", "")) > 60 else out.get("final_smiles"))
    print("  target_achieved:", out.get("target_achieved"))
    print("  total_iterations:", out.get("total_iterations"))
    print("  history length:", len(out.get("history", [])))
    if out.get("history"):
        h0 = out["history"][0]
        print("  iter0 overall:", h0.get("overall_prob"))
    print("OK: goal achieved (scientist params used, generation ran).")

if __name__ == "__main__":
    main()
