"""
実行後アウトプット生成スクリプト
使い方: python metacog/generate_outputs.py [--log metacog/logs/agent_log.jsonl] [--sim-output output]
生成物:
  - metacog/logs/report.md          報告書
  - metacog/logs/inner_life.md      心境のログ（メタ認知エージェントの内側）
  - metacog/logs/action_log.md      行動のログ（シミュレーション20体の動き）
  - output/simulation.mp4           動画（ffmpegが必要）
"""
import argparse
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def format_ts(iso):
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%H:%M:%S")
    except Exception:
        return iso


# ──────────────────────────────────────────────
# 報告書
# ──────────────────────────────────────────────
def generate_report(log_records, sim_messages, out_path):
    session_start = next((r for r in log_records if r["event_type"] == "session_start"), {})
    session_end   = next((r for r in log_records if r["event_type"] == "session_end"), {})
    thoughts      = [r for r in log_records if r["event_type"] == "thought"]
    excitements   = [r for r in log_records if r["event_type"] == "excitement"]
    diffs         = [r for r in log_records if r["event_type"] == "diff"]
    searches      = [r for r in log_records if r["event_type"] == "search"]

    total_cycles  = max((r.get("cycle", 0) for r in log_records), default=0)
    high_ex       = [e for e in excitements if e["score"] >= 8]
    synth_ex      = [e for e in excitements if e.get("seed_resonance") == "synthesis"]

    seed_counts = defaultdict(int)
    for e in excitements:
        seed_counts[e.get("seed_resonance", "unknown")] += 1

    final_prompt = session_end.get("final_prompt", "（記録なし）")
    initial_prompt = session_start.get("initial_prompt", "（記録なし）")

    lines = [
        "# 実験報告書",
        f"\n生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"セッションID: `{session_start.get('session_id', '-')}`",

        "\n---\n",
        "## 概要",
        "",
        "近未来AIインフラ都市を舞台に、20体のAIエージェントによるシミュレーションと、",
        "そのシミュレーションを観察しながら自己書き換えを続けるメタ認知エージェントを同時実行した実験。",
        "",
        "**2つのSEED（不変の問い）**",
        "- `SEED_HUMAN`: AIが社会を回す世界で、人間は何をするのか",
        "- `SEED_AI`: 情報生命はどのように死ぬのか",

        "\n---\n",
        "## 数値サマリー",
        "",
        f"| 項目 | 値 |",
        f"|---|---|",
        f"| メタ認知サイクル数 | {total_cycles} |",
        f"| Web検索回数 | {len(searches)} |",
        f"| 興奮記録数 | {len(excitements)} |",
        f"| 高興奮（score≥8）| {len(high_ex)} 件 |",
        f"| Synthesis共鳴 | {len(synth_ex)} 件 |",
        f"| prompt書き換え回数 | {len(diffs)} 回 |",
        f"| シミュレーションメッセージ数 | {len(sim_messages)} 件 |",

        "\n---\n",
        "## SEEDの共鳴分布",
        "",
        "| seed_resonance | 件数 |",
        "|---|---|",
    ]
    for k, v in sorted(seed_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| `{k}` | {v} |")

    lines += [
        "\n---\n",
        "## prompt書き換え履歴",
        "",
    ]
    for d in diffs:
        lines.append(f"### サイクル{d['cycle']} — `{d['section']}` (v{d['prompt_version']})")
        lines.append(f"**理由**: {d['reason']}")
        lines.append(f"\n```diff\n{d.get('unified_diff', '（diff記録なし）')}\n```\n")

    lines += [
        "\n---\n",
        "## 最高興奮トピック（score≥9）",
        "",
    ]
    for e in sorted(high_ex, key=lambda x: -x["score"]):
        lines.append(f"- **[score {e['score']} / {e['seed_resonance']}]** {e['topic']}")
        lines.append(f"  > {e['reason'][:150]}")

    lines += [
        "\n---\n",
        "## 最終prompt状態",
        "",
        "```",
        final_prompt[:3000],
        "```",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"報告書: {out_path}")


# ──────────────────────────────────────────────
# 心境のログ
# ──────────────────────────────────────────────
def generate_inner_life(log_records, out_path):
    cycles = defaultdict(list)
    for r in log_records:
        c = r.get("cycle")
        if c is not None:
            cycles[c].append(r)

    lines = [
        "# 心境のログ — メタ認知エージェントの内側",
        "",
        "各サイクルで何を検索し、何に心踊り、自分をどう書き換えたか。",
        "",
    ]

    for cycle_num in sorted(cycles.keys()):
        records = cycles[cycle_num]
        lines.append(f"\n---\n\n## サイクル {cycle_num}")

        sim_inp = next((r for r in records if r["event_type"] == "sim_input"), None)
        if sim_inp:
            step = sim_inp.get("sim_step", "?")
            msgs = sim_inp.get("sim_state", {}).get("messages", [])
            fires = sim_inp.get("sim_state", {}).get("active_fires", [])
            lines.append(f"\n**シミュレーション観察（step {step}）**")
            if fires:
                for fire in fires:
                    lines.append(f"- システム異常: `{fire['name']}` 強度={fire['intensity']}")
            if msgs:
                lines.append(f"- エージェントの声（直近{len(msgs)}件）:")
                for m in msgs[:3]:
                    content = m.get("message", m.get("content", ""))[:80]
                    lines.append(f"  - Agent{m.get('from','?')}: {content}")

        searches = [r for r in records if r["event_type"] == "search"]
        if searches:
            lines.append(f"\n**検索（{len(searches)}回）**")
            for s in searches:
                lines.append(f"- `{s['query']}` — {s.get('intent','')[:60]}")

        excitements = [r for r in records if r["event_type"] == "excitement"]
        for e in excitements:
            mark = "🔥" if e["score"] >= 9 else "✨" if e["score"] >= 7 else "·"
            lines.append(f"\n**興奮 {mark} score={e['score']} [{e['seed_resonance']}]**")
            lines.append(f"> {e['topic']}")
            lines.append(f"\n{e['reason'][:300]}")

        diffs = [r for r in records if r["event_type"] == "diff"]
        for d in diffs:
            lines.append(f"\n**自己書き換え → `{d['section']}` v{d['prompt_version']}**")
            lines.append(f"> {d['reason'][:200]}")
            lines.append(f"\n```\n{d['after'][:400]}\n```")

        thought = next((r for r in records if r["event_type"] == "thought"), None)
        if thought:
            lines.append(f"\n**思考の着地点**")
            lines.append(f"\n{thought['text'][:500]}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"心境ログ: {out_path}")


# ──────────────────────────────────────────────
# 行動のログ（シミュレーション20体）
# ──────────────────────────────────────────────
def generate_action_log(sim_messages, memory_records, out_path):
    by_step = defaultdict(list)
    for m in sim_messages:
        by_step[m.get("step", 0)].append(m)

    mem_by_step = defaultdict(list)
    for m in memory_records:
        mem_by_step[m.get("step", 0)].append(m)

    lines = [
        "# 行動のログ — シミュレーション20体の動き",
        "",
        "各ステップでエージェントが何を伝え、何を記憶したか。",
        "",
    ]

    all_steps = sorted(set(list(by_step.keys()) + list(mem_by_step.keys())))
    for step in all_steps:
        msgs = by_step.get(step, [])
        mems = mem_by_step.get(step, [])
        if not msgs and not mems:
            continue

        lines.append(f"\n---\n\n## Step {step}")

        if msgs:
            lines.append(f"\n**メッセージ（{len(msgs)}件）**")
            for m in msgs[:5]:
                lines.append(f"- Agent{m['from']} → Agent{m['to']}: {m.get('message','')[:100]}")
            if len(msgs) > 5:
                lines.append(f"- ... 他{len(msgs)-5}件")

        if mems:
            lines.append(f"\n**記憶・推論（{len(mems)}件）**")
            for m in mems[:3]:
                lines.append(f"- Agent{m.get('agent_id','?')}: {m.get('memory','')[:100]}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"行動ログ: {out_path}")


# ──────────────────────────────────────────────
# 動画生成
# ──────────────────────────────────────────────
def generate_video(frames_dir, out_path):
    if not os.path.exists(frames_dir):
        print(f"動画スキップ: フレームディレクトリが存在しません ({frames_dir})")
        return
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if not frames:
        print("動画スキップ: フレームが見つかりません")
        return

    gen_script = os.path.join(os.path.dirname(__file__), "..", "visualization", "generate_video.py")
    if os.path.exists(gen_script):
        result = subprocess.run(
            ["python", gen_script, frames_dir, "--output", out_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"動画: {out_path}")
        else:
            print(f"動画生成失敗: {result.stderr[:200]}")
    else:
        result = subprocess.run(
            ["ffmpeg", "-y", "-framerate", "5", "-pattern_type", "glob",
             "-i", os.path.join(frames_dir, "frame_*.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"動画: {out_path}")
        else:
            print(f"動画生成失敗（ffmpegが必要）: {result.stderr[:200]}")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="metacog/logs/agent_log.jsonl")
    parser.add_argument("--sim-output", default="output")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    log_dir = os.path.dirname(args.log)
    log_records   = load_jsonl(args.log)
    sim_messages  = load_jsonl(os.path.join(args.sim_output, "messages.jsonl"))
    memory_records = load_jsonl(os.path.join(args.sim_output, "memory_reasoning.jsonl"))

    print(f"ログ読み込み: {len(log_records)}件 / メッセージ: {len(sim_messages)}件")

    generate_report(log_records, sim_messages, os.path.join(log_dir, "report.md"))
    generate_inner_life(log_records, os.path.join(log_dir, "inner_life.md"))
    generate_action_log(sim_messages, memory_records, os.path.join(log_dir, "action_log.md"))

    if not args.skip_video:
        generate_video(args.sim_output, os.path.join(args.sim_output, "simulation.mp4"))

    print("\n完了。出力ファイル:")
    print(f"  {log_dir}/report.md")
    print(f"  {log_dir}/inner_life.md")
    print(f"  {log_dir}/action_log.md")
    print(f"  {args.sim_output}/simulation.mp4")


if __name__ == "__main__":
    main()
