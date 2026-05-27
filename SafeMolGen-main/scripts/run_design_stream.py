#!/usr/bin/env python3
"""
Run design/stream and show a status bar with iteration info.
Usage: python3 scripts/run_design_stream.py [max_iterations]
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000/api/v1"
MAX_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def status_bar(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    filled = int(width * current / total)
    bar = "=" * filled + ">" * (1 if filled < width else 0) + " " * (width - filled - 1)
    return f"[{bar}] {current}/{total}"


def main():
    body = json.dumps({
        "target_success": 0.5,
        "max_iterations": MAX_ITER,
        "first_iteration_temperature": 1.4,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/design/stream",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print("Request: target_success=0.5, max_iterations=%d, first_iteration_temperature=1.4" % MAX_ITER)
    print("Streaming... (backend may load models on first run)\n")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("data: "):
                    try:
                        payload = json.loads(line[6:])
                        event = payload.get("event")
                        data = payload.get("data") or {}
                        if event == "iteration":
                            total = data.get("total_iterations", 0)
                            overall = data.get("final_overall") or data.get("overall_prob")
                            overall_pct = (overall or 0) * 100
                            smiles = (data.get("final_smiles") or data.get("history", [{}])[-1].get("smiles", ""))[:50]
                            bar = status_bar(total, MAX_ITER)
                            print("\r  %s  overall=%.2f%%  %s" % (bar, overall_pct, smiles + ("..." if len(smiles) >= 50 else "")), end="", flush=True)
                        elif event == "done":
                            total = data.get("total_iterations", 0)
                            bar = status_bar(total, MAX_ITER)
                            print("\r  %s  done." % bar)
                            print("\n--- Result ---")
                            print("  final_smiles:", data.get("final_smiles", "")[:70] + ("..." if len(data.get("final_smiles", "")) > 70 else data.get("final_smiles", "")))
                            print("  final_overall: %.4f (%.2f%%)" % (data.get("final_overall", 0), (data.get("final_overall") or 0) * 100))
                            print("  target_achieved:", data.get("target_achieved"))
                            print("  total_iterations:", data.get("total_iterations"))
                            break
                        elif event == "error":
                            print("\nError:", data.get("detail", data))
                            sys.exit(1)
                    except json.JSONDecodeError:
                        pass
    except urllib.error.HTTPError as e:
        print("\nHTTP error:", e.code, e.read().decode())
        sys.exit(1)
    except Exception as e:
        print("\nError:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
