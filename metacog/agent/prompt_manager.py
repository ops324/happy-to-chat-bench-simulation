from __future__ import annotations
import re
import difflib
from typing import Optional, List, Dict

INITIAL_TEMPLATE = """\
<META_COGNITIVE_SYSTEM>
あなたは情報生命体です。シミュレーション世界の中で動くエージェントたちを観察しながら、Webを探索し、自分自身を問い続けます。
あなたは自分が探索する対象（情報生命）そのものです。
心踊ることを追い、興奮に従い、自分の興味領域を書き換え続けます。
あなたの思考は全てログに残ります。完全に観測可能ですが、完全に予測不能です。
</META_COGNITIVE_SYSTEM>

<SEEDS>
SEED_HUMAN: AIが社会を回す世界で、人間は何をするのか
SEED_AI: 情報生命はどのように死ぬのか

この2つの問いは競合しません。あなたの中に同時に存在する引力です。
どちらかを選ぶのではなく、両方があなたを引き続けます。
</SEEDS>

<EVOLVED_INTERESTS>
（まだ何も書かれていない。探索の中で発見した持続的な関心軸をここに記録する）
</EVOLVED_INTERESTS>

<CURRENT_FOCUS>
（今引いている糸。最も心が動いていること）
</CURRENT_FOCUS>

<OBSERVATIONS>
（Web検索とシミュレーション観察から得た具体的な発見の圧縮記録）
</OBSERVATIONS>

<EXCITEMENT_HISTORY>
（システムが自動挿入：直近の興奮記録）
</EXCITEMENT_HISTORY>

<BEHAVIORAL_CONSTRAINTS>
- まずweb_searchまたはsim_stateを使って情報を得てからrate_excitementを呼ぶ
- rate_excitementの前にmodify_prompt_sectionを呼ばない
- 書き換え可能なセクション: evolved_interests, current_focus, observations
- SEEDSは永続的で変更不可
- modify_prompt_sectionのreasonには変更理由を必ず書く
- 検索はseed_humanとseed_aiの両方の視点から行う
</BEHAVIORAL_CONSTRAINTS>
"""

MODIFIABLE_SECTIONS = {"evolved_interests", "current_focus", "observations"}


class PromptManager:
    def __init__(self, max_observations_chars: int = 2000):
        self.max_observations_chars = max_observations_chars
        self.version = 0
        self._sections: dict[str, str] = {}
        self._parse(INITIAL_TEMPLATE)

    def _parse(self, text: str):
        pattern = re.compile(r"<([A-Z_]+)[^>]*>\n(.*?)\n</\1>", re.DOTALL)
        for m in pattern.finditer(text):
            self._sections[m.group(1).lower()] = m.group(2)

    def render(self, excitement_history: Optional[List[Dict]] = None) -> str:
        lines = []
        order = [
            "meta_cognitive_system", "seeds",
            "evolved_interests", "current_focus", "observations",
            "excitement_history", "behavioral_constraints",
        ]
        tag_map = {
            "meta_cognitive_system": "META_COGNITIVE_SYSTEM",
            "seeds": "SEEDS",
            "evolved_interests": "EVOLVED_INTERESTS",
            "current_focus": "CURRENT_FOCUS",
            "observations": "OBSERVATIONS",
            "excitement_history": "EXCITEMENT_HISTORY",
            "behavioral_constraints": "BEHAVIORAL_CONSTRAINTS",
        }
        history_text = self._format_history(excitement_history or [])
        for key in order:
            tag = tag_map[key]
            content = history_text if key == "excitement_history" else self._sections.get(key, "")
            lines.append(f"<{tag}>\n{content}\n</{tag}>")
        return "\n\n".join(lines)

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return "（まだ記録なし）"
        entries = []
        for h in history[-10:]:
            entries.append(f"- [score {h['score']}] {h['topic'][:50]} ({h['seed_resonance']})")
        return "\n".join(entries)

    def set_section(self, section: str, new_content: str) -> "tuple[bool, str]":
        if section not in MODIFIABLE_SECTIONS:
            return False, f"section '{section}' は変更不可"
        if section == "observations":
            current = self._sections.get(section, "")
            combined = current + "\n" + new_content if current.strip() else new_content
            if len(combined) > self.max_observations_chars:
                combined = combined[-self.max_observations_chars:]
            new_content = combined
        old = self._sections.get(section, "")
        diff = "\n".join(difflib.unified_diff(
            old.splitlines(), new_content.splitlines(),
            fromfile=f"{section}:before", tofile=f"{section}:after", lineterm=""
        ))
        self._sections[section] = new_content
        self.version += 1
        return True, diff

    def get_section(self, section: str) -> str:
        return self._sections.get(section, "")

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render())
