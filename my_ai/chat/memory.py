"""
Application-level memory. IMPORTANT AND HONEST DISTINCTION:

This is NOT the model's knowledge. The model's "knowledge" lives in its
neural weights and only changes when we TRAIN. This module is plain
software: JSON files on disk that the chat app reads and pastes into the
prompt. The model "remembers" a fact only because the fact is literally
re-inserted into its context window on every turn.

Stores:
- conversation history (per-session JSONL)
- user-provided facts ("remember that my dog is called Rex")
- pinned important interactions

Later, selected conversations can be exported as fine-tuning data --
THAT is the moment memory would actually enter the weights.
"""
from __future__ import annotations

import json
import os
import time


class Memory:
    def __init__(self, root: str = "my_ai/chat/memory_store"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.facts_path = os.path.join(root, "facts.json")
        self.pins_path = os.path.join(root, "pinned.json")

    # ---------------- conversation history ----------------
    def session_path(self, session_id: str) -> str:
        return os.path.join(self.root, f"session_{session_id}.jsonl")

    def append_turn(self, session_id: str, role: str, text: str) -> None:
        with open(self.session_path(session_id), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "role": role, "text": text},
                               ensure_ascii=False) + "\n")

    def load_history(self, session_id: str) -> list[dict]:
        path = self.session_path(session_id)
        if not os.path.exists(path):
            return []
        turns = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        turns.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return turns

    def list_sessions(self) -> list[str]:
        return sorted(f[len("session_"):-len(".jsonl")]
                      for f in os.listdir(self.root)
                      if f.startswith("session_") and f.endswith(".jsonl"))

    # ---------------- facts ----------------
    def _read_json(self, path: str) -> list:
        if not os.path.exists(path):
            return []
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def add_fact(self, fact: str) -> None:
        facts = self._read_json(self.facts_path)
        facts.append({"t": time.time(), "fact": fact})
        with open(self.facts_path, "w", encoding="utf-8") as f:
            json.dump(facts, f, ensure_ascii=False, indent=1)

    def get_facts(self) -> list[str]:
        return [x["fact"] for x in self._read_json(self.facts_path)]

    def clear_facts(self) -> None:
        if os.path.exists(self.facts_path):
            os.remove(self.facts_path)

    # ---------------- pinned interactions ----------------
    def pin(self, user_text: str, assistant_text: str) -> None:
        pins = self._read_json(self.pins_path)
        pins.append({"t": time.time(), "user": user_text, "assistant": assistant_text})
        with open(self.pins_path, "w", encoding="utf-8") as f:
            json.dump(pins, f, ensure_ascii=False, indent=1)

    def get_pins(self) -> list[dict]:
        return self._read_json(self.pins_path)

    # ---------------- export for future fine-tuning ----------------
    def export_finetune_jsonl(self, out_path: str) -> int:
        """Dump all sessions as {"text": "<|user|>...<|assistant|>..."} lines."""
        n = 0
        with open(out_path, "w", encoding="utf-8") as out:
            for sid in self.list_sessions():
                for turn in self.load_history(sid):
                    out.write(json.dumps({"role": turn["role"], "text": turn["text"]},
                                         ensure_ascii=False) + "\n")
                    n += 1
        return n
