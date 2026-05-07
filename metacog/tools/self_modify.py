from __future__ import annotations
from typing import Optional
from metacog.agent.prompt_manager import PromptManager


class SelfModifyTool:
    def __init__(self, prompt_manager: PromptManager, persist_path: "Optional[str]" = None):
        self.pm = prompt_manager
        self.persist_path = persist_path

    def modify(self, section: str, new_content: str, reason: str) -> dict:
        before = self.pm.get_section(section)
        ok, diff = self.pm.set_section(section, new_content)
        if not ok:
            return {"success": False, "error": diff}
        after = self.pm.get_section(section)
        if self.persist_path:
            self.pm.save(self.persist_path)
        return {
            "success": True,
            "section": section,
            "before": before,
            "after": after,
            "unified_diff": diff,
            "prompt_version": self.pm.version,
        }
