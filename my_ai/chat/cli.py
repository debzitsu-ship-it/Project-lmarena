"""
Local chat interface around OUR model. Nothing leaves this machine:
no API calls, no network -- the "brain" is the checkpoint file you trained.

Flow:  user input -> memory/facts + conversation history -> prompt string
       -> tokenizer -> our Transformer -> logits -> sampling -> streamed reply.

Honesty note: until the model has been trained on a lot of text (and then
fine-tuned on instruction/chat data), replies will look like the training
corpus, not like a helpful assistant. A ~5M-param model trained on a few MB
produces locally-coherent but globally-rambling text. That is expected and
is exactly what "a small language model" means.

Run:  .venv/bin/python -m my_ai.chat.cli --checkpoint my_ai/checkpoints/best.pt \
          --tokenizer my_ai/checkpoints/tokenizer.json
Commands inside chat:  /remember <fact>   /facts   /pin   /reset   /quit
"""
from __future__ import annotations

import argparse
import sys
import time

from my_ai.chat.memory import Memory
from my_ai.inference.generate import generate
from my_ai.tokenizer.tokenizer import (EOS, SPECIAL_TOKENS, USER_TOK,
                                       ASSISTANT_TOK, load_tokenizer)
from my_ai.training.trainer import load_checkpoint, pick_device


def build_prompt_ids(tokenizer, facts: list[str], history: list[dict],
                     user_msg: str, context_length: int) -> list[int]:
    """
    Serialize memory + history + new message into token ids using the
    special role tokens, then truncate from the LEFT so the newest turns
    always fit inside the model's context window.
    """
    ids: list[int] = []
    for fact in facts:
        ids += [SPECIAL_TOKENS.index("<|system|>")] + tokenizer.encode(f"fact: {fact}\n")
    for turn in history:
        role_tok = USER_TOK if turn["role"] == "user" else ASSISTANT_TOK
        ids += [role_tok] + tokenizer.encode(turn["text"]) + [EOS]
    ids += [USER_TOK] + tokenizer.encode(user_msg) + [EOS, ASSISTANT_TOK]
    budget = context_length - 8  # leave room to start generating
    return ids[-budget:]


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with your from-scratch model (fully local).")
    ap.add_argument("--checkpoint", default="my_ai/checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="my_ai/checkpoints/tokenizer.json")
    ap.add_argument("--session", default=None, help="session id (default: timestamp)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    args = ap.parse_args()

    device = pick_device()
    model, ckpt = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    tokenizer = load_tokenizer(args.tokenizer)
    memory = Memory()
    session = args.session or time.strftime("%Y%m%d-%H%M%S")
    history = memory.load_history(session)

    print(f"[local chat] model: {model.num_parameters():,} params | device: {device} "
          f"| ckpt step {ckpt.get('step')} | session {session}")
    print("commands: /remember <fact>  /facts  /pin  /reset  /quit\n")

    last_user, last_reply = "", ""
    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_msg:
            continue
        if user_msg == "/quit":
            break
        if user_msg == "/reset":
            history = []
            print("[history cleared for this session context]")
            continue
        if user_msg.startswith("/remember "):
            memory.add_fact(user_msg[len("/remember "):])
            print("[fact stored -- note: stored in a JSON file, not in the model's weights]")
            continue
        if user_msg == "/facts":
            for f_ in memory.get_facts():
                print("  -", f_)
            continue
        if user_msg == "/pin":
            if last_reply:
                memory.pin(last_user, last_reply)
                print("[pinned last exchange]")
            continue

        # agent tools first: exact answers beat learned approximations
        from my_ai.chat.tools import try_calculator
        tool_answer = try_calculator(user_msg)
        if tool_answer is not None:
            print(f"AI: {tool_answer}")
            history.append({"role": "user", "text": user_msg})
            history.append({"role": "assistant", "text": tool_answer})
            memory.append_turn(session, "user", user_msg)
            memory.append_turn(session, "assistant", tool_answer)
            last_user, last_reply = user_msg, tool_answer
            continue

        ids = build_prompt_ids(tokenizer, memory.get_facts(), history,
                               user_msg, model.cfg.context_length)
        print("AI: ", end="", flush=True)
        pieces: list[int] = []

        def stream_cb(tok_id: int) -> None:
            piece = tokenizer.decode([tok_id])
            sys.stdout.write(piece)
            sys.stdout.flush()

        out = generate(model, ids, max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, top_k=args.top_k,
                       top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                       stop_tokens={EOS, USER_TOK, ASSISTANT_TOK}, device=device,
                       stream_callback=stream_cb)
        reply = tokenizer.decode(out)
        print()
        history.append({"role": "user", "text": user_msg})
        history.append({"role": "assistant", "text": reply})
        memory.append_turn(session, "user", user_msg)
        memory.append_turn(session, "assistant", reply)
        last_user, last_reply = user_msg, reply


if __name__ == "__main__":
    main()
