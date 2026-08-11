#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pick today's knowledge-card topic from the bank, deterministically and without repeats.

Usage:
    python pick_topic.py --bank topic_bank.json --tracker tracker.csv [--date YYYY-MM-DD]

Behavior:
    - Idempotent: if today already has a tracker row, re-print it.
    - Default (sequential): pick the first unused topic in bank order, so the
      series follows the learning path (module by module).
    - Optional (rotating): pick the first unused topic starting from a
      date-seeded index (legacy shuffle mode).
    - If all topics are used, start a new cycle from the seed position.
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path


FIELDS = ["date", "topic_id", "category", "title", "day", "status", "note_url"]


def load_rows(tracker_path: Path) -> list[dict]:
    if not tracker_path.exists():
        return []
    with tracker_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def seed_index(d: datetime.date, total: int) -> int:
    key = d.year * 10000 + d.month * 100 + d.day
    return key % total


def main() -> int:
    ap = argparse.ArgumentParser(description="AI PM daily knowledge-card topic picker")
    ap.add_argument("--bank", required=True, help="topic bank JSON path")
    ap.add_argument("--tracker", required=True, help="tracker CSV path")
    ap.add_argument("--date", default=datetime.date.today().isoformat(),
                    help="date YYYY-MM-DD (default: today)")
    ap.add_argument("--mode", choices=["sequential", "rotating"],
                    default="sequential",
                    help="selection mode (default: sequential)")
    args = ap.parse_args()

    bank_path = Path(args.bank)
    tracker_path = Path(args.tracker)
    d = datetime.date.fromisoformat(args.date)

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    total = len(bank)
    if total == 0:
        print("ERROR: topic bank is empty", file=sys.stderr)
        return 1

    rows = load_rows(tracker_path)
    today_rows = [r for r in rows if r.get("date") == args.date]
    if today_rows:
        r = today_rows[0]
        pick = {
            "id": r["topic_id"], "category": r["category"], "title": r["title"],
            "date": args.date, "day": r["day"], "status": r.get("status", "pending"),
        }
        print(json.dumps(pick, ensure_ascii=False))
        return 0

    used = {r.get("topic_id") for r in rows}
    chosen = None
    if args.mode == "rotating":
        start = seed_index(d, total)
        order = list(range(start, total)) + list(range(0, start))
        for idx in order:
            if str(bank[idx]["id"]) not in used:
                chosen = bank[idx]
                break
        if chosen is None:
            chosen = bank[order[0]]  # all used: start a new cycle
    else:
        for topic in bank:  # sequential: follow the learning path order
            if str(topic["id"]) not in used:
                chosen = topic
                break
        if chosen is None:
            chosen = bank[0]  # all used: start a new cycle

    day = max((int(r.get("day", 0)) for r in rows), default=0) + 1
    new_row = {
        "date": args.date,
        "topic_id": chosen["id"],
        "category": chosen["category"],
        "title": chosen["title"],
        "day": day,
        "status": "pending",
        "note_url": "",
    }
    with tracker_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not rows:
            writer.writeheader()
        writer.writerow(new_row)

    pick = {
        "id": chosen["id"], "category": chosen["category"], "title": chosen["title"],
        "date": args.date, "day": day, "status": "pending",
    }
    print(json.dumps(pick, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
