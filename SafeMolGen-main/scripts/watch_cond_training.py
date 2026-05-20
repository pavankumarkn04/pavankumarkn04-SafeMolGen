#!/usr/bin/env python3
"""
Continuous monitor for conditioned training with a live progress bar.
Run from project root: python3 scripts/watch_cond_training.py
Or: PYTHONPATH=. python3 scripts/watch_cond_training.py
Writes status to logs/cond_monitor_status.txt so it can be monitored externally.
"""
import re
import sys
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "cond_train_live.log"
STATUS_PATH = Path(__file__).resolve().parents[1] / "logs" / "cond_monitor_status.txt"
BAR_WIDTH = 40
PRETRAIN_EPOCHS = 10
PRETRAIN_BATCHES_PER_EPOCH = 782
PRETRAIN_TOTAL = PRETRAIN_EPOCHS * PRETRAIN_BATCHES_PER_EPOCH

# Patterns
RE_PRETRAIN = re.compile(
    r"Pretrain Epoch\s+(\d+)/(\d+):\s+[\d%]+\S*\s+\|\s*(\d+)/(\d+)\s+.*loss=([\d.]+)"
)
RE_RL = re.compile(r"RL Epoch\s+(\d+)\s+\|\s+Reward:\s+([\d.]+)\s+\|\s+Validity:\s+([\d.]+)%")
RE_PRETRAIN_EPOCH_LOSS = re.compile(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)")
RE_PHASE = re.compile(r"==========\s+(CONDITIONED PRETRAIN|PRETRAIN DONE|STARTING RL|ALL DONE)")


def draw_bar(frac: float, width: int = BAR_WIDTH) -> str:
    filled = int(width * frac) if frac <= 1.0 else width
    return "[" + "=" * filled + ">" * (1 if filled < width else 0) + " " * (width - filled - (1 if filled < width else 0)) + "]"


def _clean_line(line: bytes) -> str:
    try:
        text = line.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]?", "", text)
    text = re.sub(r"[\r\x00]", "", text)
    return text


def _update_phase(phase: str, text_clean: str) -> str:
    m = RE_PHASE.search(text_clean)
    if not m:
        return phase
    label = m.group(1)
    if "ALL DONE" in label:
        return "done"
    if "STARTING RL" in label or "PRETRAIN DONE" in label:
        return "rl"
    if "CONDITIONED PRETRAIN" in label:
        return "pretrain"
    return phase


def _write_status(path: Path, phase: str, **kwargs) -> None:
    with open(path, "w") as sf:
        sf.write(f"phase={phase}\n")
        for k, v in kwargs.items():
            sf.write(f"{k}={v}\n")


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        print(f"Log not found: {LOG_PATH}. Start training first.")
        sys.exit(1)

    phase = "pretrain"
    pretrain_epoch, pretrain_batch = 0, 0
    pretrain_loss = ""
    rl_epoch, rl_reward, rl_validity = 0, "", ""
    last_status = ""

    print("Watching conditioned training (Ctrl+C to stop). Progress bar updates live.\n")
    print("=" * 60)

    with open(LOG_PATH, "rb") as f:
        # Read to end to get latest state, then follow for new lines
        for line in f:
            text_clean = _clean_line(line)
            if not text_clean:
                continue
            phase = _update_phase(phase, text_clean)
            if phase == "pretrain":
                m = RE_PRETRAIN.search(text_clean)
                if m:
                    e, e_max = int(m.group(1)), int(m.group(2))
                    b, b_max = int(m.group(3)), int(m.group(4))
                    loss = m.group(5)
                    pretrain_epoch, pretrain_batch, pretrain_loss = e, b, loss
                    done = (e - 1) * b_max + b
                    total = e_max * b_max
                    frac = done / total if total else 0
                    pct = int(100 * frac)
                    bar = draw_bar(frac)
                    last_status = f"PRETRAIN  Epoch {e}/{e_max}  {bar}  {pct}%  ({done}/{total})  loss={loss}"
                    _write_status(STATUS_PATH, "pretrain", epoch=f"{e}/{e_max}", batch=f"{b}/{b_max}", loss=loss, progress=pct, bar=bar)
            elif phase == "rl":
                m = RE_RL.search(text_clean)
                if m:
                    rl_epoch = int(m.group(1))
                    rl_reward, rl_validity = m.group(2), m.group(3)
                    last_status = f"RL  Epoch {rl_epoch}  Reward: {rl_reward}  Validity: {rl_validity}%"
                    _write_status(STATUS_PATH, "rl", epoch=rl_epoch, reward=rl_reward, validity=rl_validity)
            elif phase == "done":
                break
        if last_status:
            print("\r" + last_status + " " * 15, end="", flush=True)
        print("\n(Live updates below)")
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            text_clean = _clean_line(line)
            if not text_clean:
                continue
            phase = _update_phase(phase, text_clean)
            if phase == "pretrain":
                m = RE_PRETRAIN.search(text_clean)
                if m:
                    e, e_max = int(m.group(1)), int(m.group(2))
                    b, b_max = int(m.group(3)), int(m.group(4))
                    loss = m.group(5)
                    pretrain_epoch, pretrain_batch, pretrain_loss = e, b, loss
                    done = (e - 1) * b_max + b
                    total = e_max * b_max
                    frac = done / total if total else 0
                    pct = int(100 * frac)
                    bar = draw_bar(frac)
                    status = f"PRETRAIN  Epoch {e}/{e_max}  {bar}  {pct}%  ({done}/{total})  loss={loss}"
                    if status != last_status:
                        print("\r" + status + " " * 10, end="", flush=True)
                        last_status = status
                    _write_status(STATUS_PATH, "pretrain", epoch=f"{e}/{e_max}", batch=f"{b}/{b_max}", loss=loss, progress=pct, bar=bar)
            elif phase == "rl":
                m = RE_RL.search(text_clean)
                if m:
                    rl_epoch = int(m.group(1))
                    rl_reward, rl_validity = m.group(2), m.group(3)
                    status = f"RL  Epoch {rl_epoch}  Reward: {rl_reward}  Validity: {rl_validity}%"
                    print("\r" + status + " " * 20, end="", flush=True)
                    last_status = status
                    _write_status(STATUS_PATH, "rl", epoch=rl_epoch, reward=rl_reward, validity=rl_validity)
                else:
                    m_el = RE_PRETRAIN_EPOCH_LOSS.search(text_clean)
                    if m_el and "Epoch" in text_clean and "Loss" in text_clean:
                        print("\n" + text_clean)
            elif phase == "done":
                print("\n\n" + "=" * 60)
                print("ALL DONE. Checkpoints: generator (cond_dim=25), generator_rl")
                _write_status(STATUS_PATH, "done")
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped watching.")
        sys.exit(0)
