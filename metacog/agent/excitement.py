class ExcitementEvaluator:
    def __init__(self, base_threshold: int = 7, cooldown_cycles: int = 3, adaptive: bool = True):
        self.base_threshold = base_threshold
        self.cooldown_cycles = cooldown_cycles
        self.adaptive = adaptive
        self.history: list[dict] = []
        self._last_modified: dict[str, int] = {}
        self._cycles_without_modification = 0

    def record(self, cycle: int, topic: str, score: int, reason: str, seed_resonance: str) -> tuple[bool, str]:
        threshold = self._effective_threshold()
        triggered = self._should_trigger(score, threshold)
        self.history.append({
            "cycle": cycle, "topic": topic, "score": score,
            "reason": reason, "seed_resonance": seed_resonance,
            "modification_triggered": triggered,
        })
        return triggered, f"score={score}, threshold={threshold}"

    def can_modify_section(self, section: str, current_cycle: int) -> tuple[bool, str]:
        last = self._last_modified.get(section)
        if last is None:
            return True, "初回"
        elapsed = current_cycle - last
        if elapsed < self.cooldown_cycles:
            return False, f"クールダウン中 ({elapsed}/{self.cooldown_cycles})"
        return True, "クールダウン解除"

    def mark_modified(self, section: str, cycle: int):
        self._last_modified[section] = cycle
        self._cycles_without_modification = 0

    def tick_no_modification(self):
        self._cycles_without_modification += 1

    def is_stagnant(self, threshold: int) -> bool:
        return self._cycles_without_modification >= threshold

    def _should_trigger(self, score: int, threshold: float) -> bool:
        if score == 10:
            return True
        return score >= threshold

    def _effective_threshold(self) -> float:
        if not self.adaptive or len(self.history) < 5:
            return self.base_threshold
        recent = self.history[-5:]
        mods = sum(1 for e in recent if e["modification_triggered"])
        if mods == 0:
            return max(5, self.base_threshold - 1)
        if mods >= 3:
            return min(9, self.base_threshold + 1)
        return self.base_threshold

    def recent_history(self, n: int = 10) -> list[dict]:
        return self.history[-n:]
