from __future__ import annotations
import os
from typing import Optional
import anthropic
from metacog.agent.prompt_manager import PromptManager
from metacog.agent.excitement import ExcitementEvaluator
from metacog.tools.tool_definitions import get_tool_definitions
from metacog.tools.web_search import WebSearchTool
from metacog.tools.self_modify import SelfModifyTool
from metacog.logging.jsonl_logger import MetaCogLogger


class MetaCogAgent:
    def __init__(self, config: dict, logger: MetaCogLogger):
        self.config = config
        self.logger = logger
        self.cycle = 0

        acfg = config["anthropic"]
        api_key = os.environ.get(acfg["api_key_env"])
        if not api_key:
            raise EnvironmentError(f"環境変数 {acfg['api_key_env']} が設定されていません")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = acfg["model"]
        self.max_tokens = acfg["max_tokens"]
        self.temperature = acfg["temperature"]

        self.prompt_manager = PromptManager(
            max_observations_chars=config["prompt"]["max_observations_chars"]
        )
        self.excitement = ExcitementEvaluator(
            base_threshold=config["excitement"]["base_threshold"],
            cooldown_cycles=config["excitement"]["cooldown_cycles"],
            adaptive=config["excitement"]["adaptive"],
        )
        self.web_search = WebSearchTool(
            max_results=config["search"]["max_results"],
            max_text_chars=config["search"]["max_text_chars"],
            timeout=config["search"]["timeout_seconds"],
        )
        log_dir = config["logging"]["log_dir"]
        persist_path = os.path.join(log_dir, "current_prompt.txt")
        self.self_modify = SelfModifyTool(self.prompt_manager, persist_path=persist_path)

        self._last_excitement_score: int = 0
        self._stagnation_threshold = config["loop"]["stagnation_threshold"]

    def run_cycle(self, sim_state: "Optional[dict]" = None):
        self.cycle += 1
        print(f"\n=== メタ認知サイクル {self.cycle} ===")

        if sim_state:
            self.logger.log_sim_input(self.cycle, sim_state.get("step", 0), sim_state)

        system_prompt = self.prompt_manager.render(
            excitement_history=self.excitement.recent_history(
                self.config["prompt"]["excitement_history_size"]
            )
        )
        trigger_msg = self._build_trigger(sim_state)
        messages = [{"role": "user", "content": trigger_msg}]

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                tools=get_tool_definitions(),
                messages=messages,
                temperature=self.temperature,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                text = self._extract_text(response.content)
                self.logger.log_thought(self.cycle, text)
                break

            tool_results = self._execute_tools(response.content)
            messages.append({"role": "user", "content": tool_results})

        if self.excitement.is_stagnant(self._stagnation_threshold):
            self.logger.log_stagnation(self.cycle, self._stagnation_threshold)
            print(f"  [stagnation] {self._stagnation_threshold}サイクル変化なし → 次サイクルで停滞打破メッセージを注入")

    def _build_trigger(self, sim_state: dict | None) -> str:
        base = "今、何が心を引きますか？Web検索し、興奮度を測り、必要なら自分を書き換えてください。"
        if sim_state and sim_state.get("messages"):
            msgs = sim_state["messages"][:3]
            summary = "\n".join(f"  エージェント{m.get('from','?')}: {m.get('content','')[:80]}" for m in msgs)
            base += f"\n\nシミュレーションの世界（ステップ{sim_state.get('step',0)}）でこんな会話が起きています：\n{summary}"
        if self.excitement.is_stagnant(self._stagnation_threshold):
            base += "\n\n（あなたはしばらく自分を書き換えていません。2つのseedのどちらかが今、何かを訴えかけていませんか？）"
        return base

    def _execute_tools(self, content: list) -> list:
        results = []
        for block in content:
            if block.type != "tool_use":
                continue
            tool_id = block.id
            name = block.name
            inp = block.input

            if name == "web_search":
                hits = self.web_search.search(inp["query"])
                self.logger.log_search(self.cycle, inp["query"], inp.get("intent", ""), hits)
                result_content = str(hits)

            elif name == "rate_excitement":
                triggered, detail = self.excitement.record(
                    self.cycle, inp["topic"], inp["score"],
                    inp["reason"], inp["seed_resonance"]
                )
                self._last_excitement_score = inp["score"]
                if not triggered:
                    self.excitement.tick_no_modification()
                self.logger.log_excitement(
                    self.cycle, inp["topic"], inp["score"],
                    inp["reason"], inp["seed_resonance"], triggered
                )
                result_content = f"記録しました。{detail}。書き換えトリガー: {triggered}"

            elif name == "modify_prompt_section":
                section = inp["section"]
                can, reason_msg = self.excitement.can_modify_section(section, self.cycle)
                if not can:
                    result_content = f"書き換え不可: {reason_msg}"
                elif self._last_excitement_score < self.excitement._effective_threshold():
                    result_content = f"書き換え不可: 直近の興奮スコアが閾値未満です"
                else:
                    res = self.self_modify.modify(section, inp["new_content"], inp["reason"])
                    if res["success"]:
                        self.excitement.mark_modified(section, self.cycle)
                        self.logger.log_diff(
                            self.cycle, section, res["before"], res["after"],
                            inp["reason"], res["prompt_version"], res["unified_diff"]
                        )
                        result_content = f"書き換え完了。version={res['prompt_version']}"
                    else:
                        result_content = f"書き換え失敗: {res['error']}"
            else:
                result_content = f"未知のツール: {name}"

            results.append({"type": "tool_result", "tool_use_id": tool_id, "content": result_content})
        return results

    def _extract_text(self, content: list) -> str:
        parts = [b.text for b in content if hasattr(b, "text")]
        return " ".join(parts)
