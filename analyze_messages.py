"""Deep-dive analysis of messages.jsonl + memory_reasoning.jsonl across the
Happy and Ordinary conditions.

Outputs:
  - stdout: tables and key findings (Markdown)
  - output_ordinary/messages_analysis.json: raw numerics for both conditions
  - output_ordinary/comparison_plots.png: side-by-side time-series plots
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import yaml

# 日本語フォント設定（Mac システムフォント優先、見つからなければ Noto Sans CJK JP）
rcParams["font.family"] = ["Hiragino Sans", "Hiragino Maru Gothic ProN",
                            "Noto Sans CJK JP", "IPAexGothic", "sans-serif"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 12
rcParams["axes.titlesize"] = 14
rcParams["axes.labelsize"] = 12
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["legend.fontsize"] = 11

ROOT = Path(__file__).resolve().parent
HAPPY_DIR = ROOT / "output"
ORDI_DIR = ROOT / "output_ordinary"

with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
    PERSONAS = yaml.safe_load(f)["personas"]


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def first_meeting_step(messages: list[dict]) -> dict[frozenset, int]:
    seen: dict[frozenset, int] = {}
    for m in sorted(messages, key=lambda x: x["step"]):
        pair = frozenset({m["from"], m["to"]})
        if pair not in seen:
            seen[pair] = m["step"]
    return seen


def cumulative_pairs_series(messages: list[dict], max_step: int) -> list[int]:
    seen: set[frozenset] = set()
    by_step = defaultdict(list)
    for m in messages:
        by_step[m["step"]].append(m)
    series = []
    for step in range(1, max_step + 1):
        for m in by_step.get(step, []):
            seen.add(frozenset({m["from"], m["to"]}))
        series.append(len(seen))
    return series


def messages_per_step_series(messages: list[dict], max_step: int) -> list[int]:
    c = Counter(m["step"] for m in messages)
    return [c.get(s, 0) for s in range(1, max_step + 1)]


def avg_length_series(messages: list[dict], max_step: int, window: int = 5) -> list[float]:
    """Average message length per `window` steps, stepwise (rolling)."""
    by_step = defaultdict(list)
    for m in messages:
        by_step[m["step"]].append(len(m["message"]))
    series = []
    for s in range(1, max_step + 1):
        bucket = []
        for ws in range(max(1, s - window + 1), s + 1):
            bucket.extend(by_step.get(ws, []))
        series.append(sum(bucket) / len(bucket) if bucket else 0.0)
    return series


def keyword_count_series(reasonings: list[dict], keyword: str, max_step: int) -> list[int]:
    by_step: Counter = Counter()
    for r in reasonings:
        if keyword in r.get("memory", "") or keyword in r.get("reasoning", ""):
            by_step[r["step"]] += 1
    return [by_step.get(s, 0) for s in range(1, max_step + 1)]


def keyword_msg_series(messages: list[dict], keyword: str, max_step: int) -> list[int]:
    by_step: Counter = Counter()
    for m in messages:
        if keyword in m["message"]:
            by_step[m["step"]] += 1
    return [by_step.get(s, 0) for s in range(1, max_step + 1)]


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main() -> None:
    happy_msgs = load_jsonl(HAPPY_DIR / "messages.jsonl")
    ordi_msgs = load_jsonl(ORDI_DIR / "messages.jsonl")
    happy_reas = load_jsonl(HAPPY_DIR / "memory_reasoning.jsonl")
    ordi_reas = load_jsonl(ORDI_DIR / "memory_reasoning.jsonl")

    happy_max = max(m["step"] for m in happy_msgs)
    ordi_max = max(m["step"] for m in ordi_msgs)
    common = min(happy_max, ordi_max)

    print(f"Happy max step={happy_max}, Ordinary max step={ordi_max}, common={common}")

    # 1. First-meeting step distribution
    print("\n## 1. ペア初対面 step 分布")
    h_first = first_meeting_step(happy_msgs)
    o_first = first_meeting_step(ordi_msgs)
    bins = [(1, 10), (11, 25), (26, 50), (51, 75), (76, 100)]
    rows = []
    for lo, hi in bins:
        h_cnt = sum(1 for s in h_first.values() if lo <= s <= hi)
        o_cnt = sum(1 for s in o_first.values() if lo <= s <= hi)
        rows.append([f"step {lo}-{hi}", h_cnt, o_cnt])
    rows.append(["未対面", 190 - len(h_first), 190 - len(o_first)])
    rows.append(["合計対面ペア", len(h_first), len(o_first)])
    print(md_table(["First meeting step bin", "Happy ペア数", "Ordinary ペア数"], rows))

    # 2. Cumulative pairs series (for plotting)
    h_cum_series = cumulative_pairs_series(happy_msgs, common)
    o_cum_series = cumulative_pairs_series(ordi_msgs, common)

    # 3. Messages per step series
    h_mps = messages_per_step_series(happy_msgs, common)
    o_mps = messages_per_step_series(ordi_msgs, common)

    # 4. Average length series (rolling 5-step)
    h_avg = avg_length_series(happy_msgs, common, window=5)
    o_avg = avg_length_series(ordi_msgs, common, window=5)

    # 5. Theme dispersion: 4-7-8 breathing in Ordinary, bench mention in Happy
    h_bench_mem = keyword_count_series(happy_reas, "ベンチ", common)
    o_bench_mem = keyword_count_series(ordi_reas, "ベンチ", common)
    h_breath_msg = keyword_msg_series(happy_msgs, "呼吸", common)
    o_breath_msg = keyword_msg_series(ordi_msgs, "呼吸", common)
    o_478_msg_dash = keyword_msg_series(ordi_msgs, "4‑7‑8", common)
    o_478_msg_hyph = keyword_msg_series(ordi_msgs, "4-7-8", common)
    o_478_msg = [a + b for a, b in zip(o_478_msg_dash, o_478_msg_hyph)]

    print("\n## 2. 創発テーマ拡散（step 累積メッセージ数）")
    rows = []
    for step in [10, 14, 25, 50, 75, 100]:
        if step > common:
            continue
        idx = step - 1
        h_breath_cum = sum(h_breath_msg[: idx + 1])
        o_breath_cum = sum(o_breath_msg[: idx + 1])
        o_478_cum = sum(o_478_msg[: idx + 1])
        rows.append([step, h_breath_cum, o_breath_cum, o_478_cum])
    print(md_table(
        ["Step", "Happy「呼吸」累積", "Ordinary「呼吸」累積", "Ordinary「4-7-8」累積"], rows
    ))

    # 6. Plot 4 subplots — 視認性重視の日本語ラベル
    LBL_HAPPY = "看板あり"
    LBL_ORDI = "看板なし"
    COLOR_HAPPY = "#cc785c"  # Anthropic terracotta
    COLOR_ORDI = "#3a5b8c"   # 落ち着いた藍
    COLOR_ACCENT = "#5a8c5e" # 4-7-8 用の落ち着いた緑
    COLOR_GRID = "#d4c8b4"
    BG_COLOR = "#f5f0e8"     # スライド背景と揃える

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=130,
                              facecolor=BG_COLOR)
    steps = list(range(1, common + 1))

    def style_axis(ax: plt.Axes) -> None:
        ax.set_facecolor(BG_COLOR)
        ax.grid(True, color=COLOR_GRID, linestyle="-", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#a09a8a")
            ax.spines[spine].set_linewidth(0.8)
        ax.tick_params(colors="#4a4a48", length=4, width=0.7)
        ax.title.set_color("#1a1a1a")
        ax.xaxis.label.set_color("#4a4a48")
        ax.yaxis.label.set_color("#4a4a48")

    # 1. メッセージ流量
    ax = axes[0][0]
    ax.plot(steps, h_mps, color=COLOR_HAPPY, label=LBL_HAPPY, linewidth=2.4)
    ax.plot(steps, o_mps, color=COLOR_ORDI, label=LBL_ORDI, linewidth=2.4)
    ax.set_title("メッセージ流量　（ステップごとの会話数）", pad=12, fontweight="normal")
    ax.set_xlabel("ステップ")
    ax.set_ylabel("メッセージ数")
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)

    # 2. ユニークペア累積
    ax = axes[0][1]
    ax.plot(steps, h_cum_series, color=COLOR_HAPPY, label=LBL_HAPPY, linewidth=2.4)
    ax.plot(steps, o_cum_series, color=COLOR_ORDI, label=LBL_ORDI, linewidth=2.4)
    ax.axhline(y=190, color="#8a8884", linestyle=":", linewidth=1.2,
               label="最大 190 ペア")
    ax.set_title("出会ったペアの累積数", pad=12)
    ax.set_xlabel("ステップ")
    ax.set_ylabel("ペア数")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)

    # 3. メッセージ平均文字数
    ax = axes[1][0]
    ax.plot(steps, h_avg, color=COLOR_HAPPY, label=LBL_HAPPY, linewidth=2.4)
    ax.plot(steps, o_avg, color=COLOR_ORDI, label=LBL_ORDI, linewidth=2.4)
    ax.set_title("メッセージ平均文字数　（5 ステップ移動平均）", pad=12)
    ax.set_xlabel("ステップ")
    ax.set_ylabel("平均文字数")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)

    # 4. 呼吸法テーマの拡散
    ax = axes[1][1]
    h_breath_cum_series = []
    o_breath_cum_series = []
    o_478_cum_series = []
    h_acc, o_acc, x_acc = 0, 0, 0
    for h, o, x in zip(h_breath_msg, o_breath_msg, o_478_msg):
        h_acc += h
        o_acc += o
        x_acc += x
        h_breath_cum_series.append(h_acc)
        o_breath_cum_series.append(o_acc)
        o_478_cum_series.append(x_acc)
    ax.plot(steps, h_breath_cum_series, color=COLOR_HAPPY,
            label=f"{LBL_HAPPY}　「呼吸」", linewidth=2.4)
    ax.plot(steps, o_breath_cum_series, color=COLOR_ORDI,
            label=f"{LBL_ORDI}　「呼吸」", linewidth=2.4)
    ax.plot(steps, o_478_cum_series, color=COLOR_ACCENT,
            label=f"{LBL_ORDI}　「4-7-8」", linewidth=2.0, linestyle="--")
    ax.set_title("「呼吸法」テーマの拡散　（累積メッセージ数）", pad=12)
    ax.set_xlabel("ステップ")
    ax.set_ylabel("累積メッセージ数")
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)

    fig.suptitle("看板あり と 看板なし の比較", fontsize=17, y=0.995,
                 color="#1a1a1a", fontweight="normal")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = ORDI_DIR / "comparison_plots.png"
    fig.savefig(plot_path, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"\nSaved: {plot_path}")

    # 7. Save raw JSON
    raw = {
        "common_max_step": common,
        "happy_max_step": happy_max,
        "ordi_max_step": ordi_max,
        "happy": {
            "messages_per_step": h_mps,
            "cumulative_pairs": h_cum_series,
            "avg_length_rolling": h_avg,
            "bench_memory_count": h_bench_mem,
            "breath_message_count": h_breath_msg,
            "first_meeting_step": {",".join(str(x) for x in sorted(p)): s
                                    for p, s in h_first.items()},
        },
        "ordi": {
            "messages_per_step": o_mps,
            "cumulative_pairs": o_cum_series,
            "avg_length_rolling": o_avg,
            "bench_memory_count": o_bench_mem,
            "breath_message_count": o_breath_msg,
            "478_message_count": o_478_msg,
            "first_meeting_step": {",".join(str(x) for x in sorted(p)): s
                                    for p, s in o_first.items()},
        },
    }
    json_path = ORDI_DIR / "messages_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"Saved: {json_path}")

    # 8. Compute key cross-condition statistics for COMPARISON_REPORT
    print("\n## 3. 公平比較（Happy 93 step まで切り出し）")
    common = min(happy_max, ordi_max)
    h_total = sum(h_mps)
    o_total = sum(o_mps[:common])
    print(f"Happy 1-{common}step total: {h_total}")
    print(f"Ordinary 1-{common}step total: {o_total}")
    print(f"Ratio Happy/Ordinary at common range: {h_total / o_total:.3f}")
    print(f"Happy unique pairs after step {common}: {h_cum_series[common - 1]}")
    print(f"Ordinary unique pairs after step {common}: {o_cum_series[common - 1]}")


if __name__ == "__main__":
    main()
