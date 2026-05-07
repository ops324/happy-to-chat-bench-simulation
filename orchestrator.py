"""
オーケストレーター: シミュレーション（Ollama）とメタ認知エージェント（Claude API）を同一ループで実行する。
起動: python orchestrator.py [--sim-config config.yaml] [--meta-config metacog/config.yaml] [--duration N] [--trigger N]
"""
import argparse
import logging
import uuid
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from simulation import Simulation
from metacog.agent.meta_agent import MetaCogAgent
from metacog.logging.jsonl_logger import MetaCogLogger


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main():
    parser = argparse.ArgumentParser(description="シミュレーション + メタ認知エージェント 同時実行")
    parser.add_argument("--sim-config", default="config.yaml")
    parser.add_argument("--meta-config", default="metacog/config.yaml")
    parser.add_argument("--duration", type=int, default=None, help="シミュレーションステップ数（config上書き）")
    parser.add_argument("--trigger", type=int, default=None, help="メタ認知トリガー間隔（Nステップごと）")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("orchestrator")

    with open(args.meta_config, "r", encoding="utf-8") as f:
        meta_config = yaml.safe_load(f)

    trigger_interval = args.trigger or meta_config["orchestrator"]["meta_trigger_every_n_steps"]

    sim = Simulation(config_path=args.sim_config)
    if args.duration:
        sim.duration = args.duration

    session_id = uuid.uuid4().hex[:8]
    log_dir = meta_config["logging"]["log_dir"]
    meta_logger = MetaCogLogger(log_dir=log_dir, session_id=session_id, console=meta_config["logging"]["console"])
    meta_agent = MetaCogAgent(config=meta_config, logger=meta_logger)

    meta_logger.log_session_start(
        initial_prompt=meta_agent.prompt_manager.render(),
        config={"sim_config": args.sim_config, "trigger_interval": trigger_interval, "duration": sim.duration},
    )

    if not sim.llm_client.check_connection():
        logger.error("Ollamaに接続できません。Ollamaが起動しているか確認してください。")
        return
    if not sim.llm_client.check_model_exists():
        logger.error(f"モデル '{sim.llm_client.model}' がOllamaに存在しません。")
        return

    sim.initialize_agents()
    logger.info(f"シミュレーション開始 (duration={sim.duration}, trigger={trigger_interval}ステップごと)")
    logger.info(f"メタ認知セッション: {session_id}")

    try:
        while sim.step < sim.duration:
            sim.step_simulation()
            logger.info(f"[sim] step {sim.step}/{sim.duration} 完了")

            if sim.step % trigger_interval == 0:
                sim_state = sim.get_current_state()
                meta_agent.run_cycle(sim_state=sim_state)

        logger.info("シミュレーション完了")

    except KeyboardInterrupt:
        logger.info("中断されました")
    finally:
        meta_logger.log_session_end(final_prompt=meta_agent.prompt_manager.render())
        logger.info(f"ログ保存先: {log_dir}/agent_log.jsonl")


if __name__ == "__main__":
    main()
