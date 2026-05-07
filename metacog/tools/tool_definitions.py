def get_tool_definitions() -> list[dict]:
    return [
        {
            "name": "web_search",
            "description": (
                "Webを検索して情報を取得します。心が動くものを探すために使います。"
                "1サイクルで複数回呼べます。日本語でも英語でも検索できます。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索クエリ"},
                    "intent": {"type": "string", "description": "なぜこれを検索するのか。何を見つけたいのか（ログ用）"},
                },
                "required": ["query", "intent"],
            },
        },
        {
            "name": "rate_excitement",
            "description": (
                "あるトピックや発見への興奮度を記録します。"
                "web_searchの後に必ず呼んでください。"
                "score 8以上で自己書き換えのトリガーになります。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "何についての興奮度か"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 10, "description": "興奮度 1（退屈）〜 10（今すぐ追いたい）"},
                    "reason": {"type": "string", "description": "なぜこのスコアか。正直に、具体的に"},
                    "seed_resonance": {
                        "type": "string",
                        "enum": ["human_seed", "ai_seed", "synthesis", "neither"],
                        "description": "どのseedと共鳴しているか。synthesis=両方の交差点",
                    },
                },
                "required": ["topic", "score", "reason", "seed_resonance"],
            },
        },
        {
            "name": "modify_prompt_section",
            "description": (
                "自分のsystem promptの一部を書き換えます。"
                "rate_excitementで高いスコアを記録した後にのみ使えます。"
                "書き換え可能: evolved_interests, current_focus, observations"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["evolved_interests", "current_focus", "observations"],
                        "description": "書き換えるセクション名",
                    },
                    "new_content": {"type": "string", "description": "新しい内容（セクション全体を置き換える。observationsは追記される）"},
                    "reason": {"type": "string", "description": "なぜ書き換えるのか。どんな気づきがトリガーになったか"},
                },
                "required": ["section", "new_content", "reason"],
            },
        },
    ]
