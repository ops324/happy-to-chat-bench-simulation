"""Cross-condition comparison: Happy-to-Chat bench vs Ordinary bench.

Reads both conditions' messages.jsonl and memory_reasoning.jsonl, then prints
side-by-side comparison tables in Markdown. Output is meant to be pasted into
COMPARISON_REPORT.md.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
HAPPY_DIR = ROOT / "output"
ORDI_DIR = ROOT / "output_ordinary"

with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
    PERSONAS = yaml.safe_load(f)["personas"]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_step_counts(messages: list[dict]) -> Counter:
    return Counter(m["step"] for m in messages)


def sender_receiver_counts(messages: list[dict]) -> tuple[Counter, Counter]:
    sent = Counter(m["from"] for m in messages)
    recv = Counter(m["to"] for m in messages)
    return sent, recv


def unique_pairs(messages: list[dict]) -> set[frozenset]:
    return {frozenset({m["from"], m["to"]}) for m in messages}


def central_keyword_count(reasonings: list[dict], keyword: str) -> Counter:
    """Count, per step, how many memory texts mention the central place keyword."""
    c: Counter = Counter()
    for r in reasonings:
        if keyword in r.get("memory", "") or keyword in r.get("reasoning", ""):
            c[r["step"]] += 1
    return c


def cumulative_unique_pairs(messages: list[dict]) -> dict[int, int]:
    seen: set[frozenset] = set()
    result: dict[int, int] = {}
    by_step = defaultdict(list)
    for m in messages:
        by_step[m["step"]].append(m)
    for step in sorted(by_step):
        for m in by_step[step]:
            seen.add(frozenset({m["from"], m["to"]}))
        result[step] = len(seen)
    return result


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main() -> None:
    print("=" * 72)
    print("Comparison: Happy-to-Chat Bench vs Ordinary Bench")
    print("=" * 72)

    happy_msgs = load_jsonl(HAPPY_DIR / "messages.jsonl")
    ordi_msgs = load_jsonl(ORDI_DIR / "messages.jsonl")
    happy_reas = load_jsonl(HAPPY_DIR / "memory_reasoning.jsonl")
    ordi_reas = load_jsonl(ORDI_DIR / "memory_reasoning.jsonl")

    print(f"\nHappy:    messages={len(happy_msgs)}, reasoning={len(happy_reas)}")
    print(f"Ordinary: messages={len(ordi_msgs)}, reasoning={len(ordi_reas)}")

    # 1. Messages per step
    print("\n## メッセージ流量（step 別）\n")
    h_per_step = per_step_counts(happy_msgs)
    o_per_step = per_step_counts(ordi_msgs)
    rows = []
    for step in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        rows.append([step, h_per_step.get(step, 0), o_per_step.get(step, 0),
                     o_per_step.get(step, 0) - h_per_step.get(step, 0)])
    print(md_table(["Step", "Happy", "Ordinary", "Diff (O-H)"], rows))
    h_total = sum(h_per_step.values())
    o_total = sum(o_per_step.values())
    print(f"\nTotal: Happy={h_total}, Ordinary={o_total}, "
          f"ratio Ordinary/Happy={o_total / h_total:.3f}")

    # 2. Cumulative unique pairs
    print("\n## ユニーク対話ペア累積数（step 別）\n")
    h_cum = cumulative_unique_pairs(happy_msgs)
    o_cum = cumulative_unique_pairs(ordi_msgs)
    rows = []
    for step in [1, 10, 25, 50, 75, 100]:
        h_v = h_cum.get(step) or h_cum[max(s for s in h_cum if s <= step)]
        o_v = o_cum.get(step) or o_cum[max(s for s in o_cum if s <= step)]
        rows.append([step, h_v, o_v])
    print(md_table(["Step", "Happy ペア累積", "Ordinary ペア累積"], rows))
    h_pairs = len(unique_pairs(happy_msgs))
    o_pairs = len(unique_pairs(ordi_msgs))
    max_possible = 20 * 19 // 2
    print(f"\nTotal unique pairs: Happy={h_pairs}/{max_possible} "
          f"({100 * h_pairs / max_possible:.1f}%), "
          f"Ordinary={o_pairs}/{max_possible} ({100 * o_pairs / max_possible:.1f}%)")

    # 3. Average message length per step bucket
    print("\n## メッセージ平均文字数（step 帯別）\n")
    def avg_len_bucket(messages: list[dict], lo: int, hi: int) -> float:
        bucket = [len(m["message"]) for m in messages if lo <= m["step"] <= hi]
        return sum(bucket) / len(bucket) if bucket else 0.0
    rows = []
    for lo, hi in [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]:
        rows.append([f"{lo}-{hi}",
                     f"{avg_len_bucket(happy_msgs, lo, hi):.1f}",
                     f"{avg_len_bucket(ordi_msgs, lo, hi):.1f}"])
    print(md_table(["Step range", "Happy 平均文字数", "Ordinary 平均文字数"], rows))

    # 4. Central place keyword references in reasoning
    print("\n## 中央場所への言及回数（memory + reasoning 文字列マッチ）\n")
    h_bench = central_keyword_count(happy_reas, "ベンチ")
    o_bench = central_keyword_count(ordi_reas, "ベンチ")
    h_fountain = central_keyword_count(happy_reas, "噴水")
    o_fountain = central_keyword_count(ordi_reas, "噴水")
    print(md_table(
        ["条件", "「ベンチ」言及合計", "「噴水」言及合計"],
        [["Happy", sum(h_bench.values()), sum(h_fountain.values())],
         ["Ordinary", sum(o_bench.values()), sum(o_fountain.values())]],
    ))

    # 5. Persona ranking comparison (top 5 senders, top 5 receivers per condition)
    print("\n## ペルソナ別 送信ランキング（上位5）\n")
    h_sent, h_recv = sender_receiver_counts(happy_msgs)
    o_sent, o_recv = sender_receiver_counts(ordi_msgs)
    def persona_label(i: int) -> str:
        return f"{i}: {PERSONAS[i]['name']}"
    rows = []
    h_top = h_sent.most_common(5)
    o_top = o_sent.most_common(5)
    for r in range(5):
        h_a, h_n = h_top[r]
        o_a, o_n = o_top[r]
        rows.append([r + 1, f"{persona_label(h_a)} ({h_n})",
                     f"{persona_label(o_a)} ({o_n})"])
    print(md_table(["Rank", "Happy 送信", "Ordinary 送信"], rows))

    print("\n## ペルソナ別 受信ランキング（上位5）\n")
    rows = []
    h_top = h_recv.most_common(5)
    o_top = o_recv.most_common(5)
    for r in range(5):
        h_a, h_n = h_top[r]
        o_a, o_n = o_top[r]
        rows.append([r + 1, f"{persona_label(h_a)} ({h_n})",
                     f"{persona_label(o_a)} ({o_n})"])
    print(md_table(["Rank", "Happy 受信", "Ordinary 受信"], rows))

    # 6. Strongest pairs in each condition
    print("\n## 強固な対話ペア（送受信合計 上位5）\n")
    def pair_totals(messages: list[dict]) -> Counter:
        c: Counter = Counter()
        for m in messages:
            c[frozenset({m["from"], m["to"]})] += 1
        return c
    h_pair = pair_totals(happy_msgs).most_common(5)
    o_pair = pair_totals(ordi_msgs).most_common(5)
    rows = []
    for r in range(5):
        a1, a2 = sorted(h_pair[r][0])
        b1, b2 = sorted(o_pair[r][0])
        rows.append([
            r + 1,
            f"{a1}↔{a2}: {h_pair[r][1]}通 ({PERSONAS[a1]['name']} ↔ {PERSONAS[a2]['name']})",
            f"{b1}↔{b2}: {o_pair[r][1]}通 ({PERSONAS[b1]['name']} ↔ {PERSONAS[b2]['name']})",
        ])
    print(md_table(["Rank", "Happy ペア", "Ordinary ペア"], rows))

    # 7. Isolated agents (low send + low recv)
    print("\n## 孤立傾向のエージェント（送信 + 受信 < 50）\n")
    rows = []
    for cond, sent, recv in [("Happy", h_sent, h_recv), ("Ordinary", o_sent, o_recv)]:
        for i in range(20):
            total = sent.get(i, 0) + recv.get(i, 0)
            if total < 50:
                rows.append([cond, persona_label(i), sent.get(i, 0), recv.get(i, 0)])
    if rows:
        print(md_table(["条件", "ペルソナ", "送信", "受信"], rows))
    else:
        print("(該当なし — 全エージェントが 50 通以上交信)")

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
