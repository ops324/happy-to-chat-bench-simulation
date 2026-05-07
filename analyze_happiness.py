"""Lexicon-based happiness analysis for the bench-comparison study.

Estimates per-agent happiness from the natural language of `memory` and
`reasoning` fields by counting positive vs. negative emotion words. Output is
a per-agent and per-condition score, used as a proxy for subjective wellbeing.

Usage:
    python analyze_happiness.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
HAPPY_DIR = ROOT / "output"
ORDI_DIR = ROOT / "output_ordinary"

with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
    PERSONAS = yaml.safe_load(f)["personas"]

# 日本語感情語彙。memory/reasoning に頻出する自然な表現に絞った
POSITIVE_WORDS = [
    "嬉しい", "楽しい", "楽しみ", "心地よい", "心地良い", "リラックス",
    "落ち着", "安心", "穏やか", "温かい", "あたたか", "和や", "和む",
    "幸せ", "感謝", "ありがた", "嬉しく", "うれしい", "面白い", "おもしろい",
    "好き", "気持ちいい", "気持ちよい", "癒さ", "癒や", "微笑", "ほっと",
    "やさしい", "優しい", "穏や", "豊か", "充実", "満たさ", "満足",
    "誇らし", "わくわく", "ワクワク", "晴れやか", "明るい", "弾む", "弾ん",
    "感動", "嬉しそう", "笑顔", "ほほえ", "ほほえま", "親しみ", "親しく",
    "繋がり", "つながり", "共有", "共感", "賑わ", "賑やか", "にぎや",
]

NEGATIVE_WORDS = [
    "疲れ", "つかれ", "不安", "寂しい", "さびし", "孤独", "孤立",
    "緊張", "焦り", "焦って", "心配", "怖い", "こわい", "辛い", "つらい",
    "苦し", "痛い", "悲しい", "かなし", "落ち込", "気が引け", "気まず",
    "退屈", "つまらな", "煩わし", "うるさ", "騒が", "騒々", "うんざり",
    "億劫", "おっくう", "ため息", "嫌", "いや", "気が進ま", "避けたい",
    "避け", "逃げ", "閉じこも", "ひきこも", "怯え", "おびえ", "静かに過ごし",
    "じっくり過ご", "周囲に人が少な",
]


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def score_text(text: str) -> tuple[int, int]:
    pos = sum(text.count(w) for w in POSITIVE_WORDS)
    neg = sum(text.count(w) for w in NEGATIVE_WORDS)
    return pos, neg


def per_agent_scores(reasonings: list[dict]) -> dict[int, dict]:
    """Returns per-agent aggregated stats: positive count, negative count,
    net (pos - neg), per-step normalized."""
    by_agent: dict[int, dict] = defaultdict(lambda: {"pos": 0, "neg": 0, "n": 0})
    for r in reasonings:
        text = r.get("memory", "") + " " + r.get("reasoning", "")
        pos, neg = score_text(text)
        a = by_agent[r["id"]]
        a["pos"] += pos
        a["neg"] += neg
        a["n"] += 1
    result = {}
    for aid, a in by_agent.items():
        n = max(1, a["n"])
        result[aid] = {
            "pos": a["pos"],
            "neg": a["neg"],
            "net": a["pos"] - a["neg"],
            "pos_per_step": a["pos"] / n,
            "neg_per_step": a["neg"] / n,
            "net_per_step": (a["pos"] - a["neg"]) / n,
            "n_steps": a["n"],
        }
    return result


def overall_average(scores: dict[int, dict]) -> dict:
    n = len(scores)
    if n == 0:
        return {}
    return {
        "mean_pos_per_step": sum(s["pos_per_step"] for s in scores.values()) / n,
        "mean_neg_per_step": sum(s["neg_per_step"] for s in scores.values()) / n,
        "mean_net_per_step": sum(s["net_per_step"] for s in scores.values()) / n,
        "total_pos": sum(s["pos"] for s in scores.values()),
        "total_neg": sum(s["neg"] for s in scores.values()),
    }


def fmt_pct_change(a: float, b: float) -> str:
    if a == 0:
        return "—"
    return f"{(b - a) / abs(a) * 100:+.1f}%"


def main() -> None:
    happy_reas = load_jsonl(HAPPY_DIR / "memory_reasoning.jsonl")
    ordi_reas = load_jsonl(ORDI_DIR / "memory_reasoning.jsonl")

    h_scores = per_agent_scores(happy_reas)
    o_scores = per_agent_scores(ordi_reas)

    h_avg = overall_average(h_scores)
    o_avg = overall_average(o_scores)

    print("=" * 64)
    print("幸福度（語彙ベース）比較")
    print("=" * 64)

    print("\n## 全体平均（1 ステップあたり）")
    print(f"            ポジティブ語  ネガティブ語  ネット")
    print(f"看板あり:   {h_avg['mean_pos_per_step']:.3f}        "
          f"{h_avg['mean_neg_per_step']:.3f}        {h_avg['mean_net_per_step']:+.3f}")
    print(f"看板なし:   {o_avg['mean_pos_per_step']:.3f}        "
          f"{o_avg['mean_neg_per_step']:.3f}        {o_avg['mean_net_per_step']:+.3f}")

    print(f"\n総ポジティブ語: 看板あり={h_avg['total_pos']}, 看板なし={o_avg['total_pos']}")
    print(f"総ネガティブ語: 看板あり={h_avg['total_neg']}, 看板なし={o_avg['total_neg']}")

    print("\n## ペルソナ別（ネット値: 1 ステップあたり）")
    print(f"{'Agent':>5}  {'ペルソナ':<22}  {'看板あり':>10}  {'看板なし':>10}  {'差':>10}")
    rows_for_json = []
    for i in range(20):
        h_net = h_scores.get(i, {}).get("net_per_step", 0)
        o_net = o_scores.get(i, {}).get("net_per_step", 0)
        diff = o_net - h_net
        name = PERSONAS[i]["name"]
        print(f"{i:>5}  {name:<22}  {h_net:>+10.3f}  {o_net:>+10.3f}  {diff:>+10.3f}")
        rows_for_json.append({
            "id": i,
            "name": name,
            "happy_net_per_step": h_net,
            "ordi_net_per_step": o_net,
            "diff": diff,
            "happy_pos": h_scores.get(i, {}).get("pos", 0),
            "happy_neg": h_scores.get(i, {}).get("neg", 0),
            "ordi_pos": o_scores.get(i, {}).get("pos", 0),
            "ordi_neg": o_scores.get(i, {}).get("neg", 0),
        })

    # Sort by largest absolute change
    print("\n## 最も変化の大きいペルソナ（差 = 看板なし − 看板あり）")
    sorted_rows = sorted(rows_for_json, key=lambda r: r["diff"])
    print("\n看板なしで幸福度が下がった上位 3:")
    for r in sorted_rows[:3]:
        print(f"  Agent {r['id']:2d} ({r['name']}): "
              f"{r['happy_net_per_step']:+.3f} → {r['ordi_net_per_step']:+.3f}  "
              f"({r['diff']:+.3f})")

    print("\n看板なしで幸福度が上がった上位 3:")
    for r in sorted_rows[-3:][::-1]:
        print(f"  Agent {r['id']:2d} ({r['name']}): "
              f"{r['happy_net_per_step']:+.3f} → {r['ordi_net_per_step']:+.3f}  "
              f"({r['diff']:+.3f})")

    # 特に注目: Agent 11 (サラリーマン) — 既存比較で取り上げ済み
    print("\n## ハイライト（既存比較ですでに注目したエージェント）")
    for aid in [11, 3, 12, 18]:
        r = rows_for_json[aid]
        print(f"  Agent {aid:2d} ({r['name']}): "
              f"看板あり {r['happy_net_per_step']:+.3f} ／ 看板なし {r['ordi_net_per_step']:+.3f}")

    # Save raw data
    out = {
        "overall": {"happy": h_avg, "ordi": o_avg},
        "per_agent": rows_for_json,
        "lexicon": {
            "positive_size": len(POSITIVE_WORDS),
            "negative_size": len(NEGATIVE_WORDS),
        },
    }
    json_path = ORDI_DIR / "happiness_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
