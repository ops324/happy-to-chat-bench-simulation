import json
import os
from datetime import datetime, timezone


class MetaCogLogger:
    def __init__(self, log_dir: str, session_id: str, console: bool = True):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "agent_log.jsonl")
        self.session_id = session_id
        self.console = console

    def _write(self, record: dict):
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        record["session_id"] = self.session_id
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.console:
            t = record["event_type"]
            cycle = record.get("cycle", "-")
            if t == "search":
                print(f"  [cycle {cycle}] search: {record['query'][:60]}")
            elif t == "excitement":
                print(f"  [cycle {cycle}] excitement: score={record['score']} topic={record['topic'][:40]}")
            elif t == "diff":
                print(f"  [cycle {cycle}] diff: section={record['section']} v{record['prompt_version']}")
            elif t == "thought":
                print(f"  [cycle {cycle}] thought: {record['text'][:80]}...")
            elif t == "sim_input":
                print(f"  [cycle {cycle}] sim_input: step={record['sim_step']}")
            elif t in ("session_start", "session_end"):
                print(f"[{t}] session={self.session_id}")

    def log_session_start(self, initial_prompt: str, config: dict):
        self._write({"event_type": "session_start", "initial_prompt": initial_prompt, "config": config})

    def log_session_end(self, final_prompt: str):
        self._write({"event_type": "session_end", "final_prompt": final_prompt})

    def log_search(self, cycle: int, query: str, intent: str, results: list):
        self._write({"event_type": "search", "cycle": cycle, "query": query, "intent": intent, "results": results})

    def log_excitement(self, cycle: int, topic: str, score: int, reason: str, seed_resonance: str, modification_triggered: bool):
        self._write({
            "event_type": "excitement", "cycle": cycle,
            "topic": topic, "score": score, "reason": reason,
            "seed_resonance": seed_resonance, "modification_triggered": modification_triggered,
        })

    def log_diff(self, cycle: int, section: str, before: str, after: str, reason: str, prompt_version: int, unified_diff: str):
        self._write({
            "event_type": "diff", "cycle": cycle,
            "section": section, "before": before, "after": after,
            "reason": reason, "prompt_version": prompt_version, "unified_diff": unified_diff,
        })

    def log_thought(self, cycle: int, text: str):
        self._write({"event_type": "thought", "cycle": cycle, "text": text})

    def log_sim_input(self, cycle: int, sim_step: int, sim_state: dict):
        self._write({"event_type": "sim_input", "cycle": cycle, "sim_step": sim_step, "sim_state": sim_state})

    def log_stagnation(self, cycle: int, cycles_without_modification: int):
        self._write({"event_type": "stagnation", "cycle": cycle, "cycles_without_modification": cycles_without_modification})
