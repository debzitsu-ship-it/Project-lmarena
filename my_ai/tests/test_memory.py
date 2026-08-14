"""Memory + chat prompt tests. Run: .venv/bin/pytest my_ai/tests/test_memory.py -q"""
from my_ai.chat.cli import build_prompt_ids
from my_ai.chat.memory import Memory
from my_ai.tokenizer.tokenizer import BPETokenizer, N_SPECIAL, USER_TOK, ASSISTANT_TOK


def test_memory_roundtrip(tmp_path):
    mem = Memory(root=str(tmp_path))
    mem.append_turn("s1", "user", "hello")
    mem.append_turn("s1", "assistant", "hi there")
    hist = mem.load_history("s1")
    assert [t["role"] for t in hist] == ["user", "assistant"]
    assert hist[0]["text"] == "hello"

    mem.add_fact("my dog is called Rex")
    assert mem.get_facts() == ["my dog is called Rex"]

    mem.pin("q", "a")
    assert mem.get_pins()[0]["user"] == "q"
    assert mem.list_sessions() == ["s1"]


def test_export_finetune(tmp_path):
    mem = Memory(root=str(tmp_path))
    mem.append_turn("s1", "user", "a")
    mem.append_turn("s1", "assistant", "b")
    out = str(tmp_path / "ft.jsonl")
    assert mem.export_finetune_jsonl(out) == 2


def test_prompt_truncates_from_left(tmp_path):
    tok = BPETokenizer()
    tok.train("hello world how are you today " * 30, vocab_size=N_SPECIAL + 256 + 20)
    history = [{"role": "user", "text": "x" * 500}, {"role": "assistant", "text": "y" * 500}]
    ids = build_prompt_ids(tok, ["fact one"], history, "newest question", context_length=128)
    assert len(ids) <= 120
    # the newest message must survive truncation
    assert ASSISTANT_TOK == ids[-1]
    tail_text = tok.decode(ids[-40:])
    assert "newest question" in tail_text
