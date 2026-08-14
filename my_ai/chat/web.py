"""
Minimal local web chat UI around OUR from-scratch model.

Same honesty guarantees as the CLI:
- The "brain" is my_ai/checkpoints/*.pt -- weights we trained ourselves.
- NO external AI API is called. The only network traffic is between your
  browser and this little Python server; generation happens locally.
- Uses only the Python standard library (http.server) -- no web framework.

Run:  .venv/bin/python -m my_ai.chat.web
Then open http://localhost:8080 in a browser.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from my_ai.chat.cli import build_prompt_ids
from my_ai.chat.memory import Memory
from my_ai.inference.generate import generate
from my_ai.tokenizer.tokenizer import EOS, USER_TOK, ASSISTANT_TOK, load_tokenizer
from my_ai.training.trainer import load_checkpoint, pick_device

HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>my_ai — local chat</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:#0f1117; color:#e6e6e6;
         display:flex; flex-direction:column; height:100vh; }
  header { padding:12px 18px; background:#161a23; border-bottom:1px solid #262b38; }
  header h1 { font-size:16px; margin:0; }
  header p { margin:4px 0 0; font-size:12px; color:#8a93a6; }
  #log { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:75%; padding:10px 14px; border-radius:14px; line-height:1.45; white-space:pre-wrap; }
  .user { align-self:flex-end; background:#2b5cd9; color:#fff; border-bottom-right-radius:4px; }
  .ai   { align-self:flex-start; background:#1e2330; border:1px solid #2b3245; border-bottom-left-radius:4px; }
  .sys  { align-self:center; font-size:12px; color:#8a93a6; }
  form { display:flex; gap:8px; padding:14px 18px; background:#161a23; border-top:1px solid #262b38; }
  input { flex:1; padding:12px 14px; border-radius:10px; border:1px solid #2b3245;
          background:#0f1117; color:#e6e6e6; font-size:15px; outline:none; }
  input:focus { border-color:#2b5cd9; }
  button { padding:12px 22px; border-radius:10px; border:0; background:#2b5cd9; color:#fff;
           font-size:15px; cursor:pointer; }
  button:disabled { opacity:.5; cursor:wait; }
</style>
</head>
<body>
<header>
  <h1>my_ai — chat with your from-scratch model</h1>
  <p id="info">loading…</p>
</header>
<div id="log">
  <div class="msg sys">This model was trained from random weights on this machine.
  It only "knows" its tiny training set (trains, cats, the sun, greetings, Alice in Wonderland…).
  Off-topic questions will produce rambling — that is honest small-model behavior, not a bug.
  Tip: type <b>/remember your fact</b> to store a fact in local memory.</div>
</div>
<form id="f">
  <input id="box" placeholder="Type a message, e.g.  Explain trains." autocomplete="off" autofocus>
  <button id="send" type="submit">Send</button>
</form>
<script>
const log = document.getElementById('log');
const box = document.getElementById('box');
const send = document.getElementById('send');

function add(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

fetch('info').then(r => r.json()).then(j => {
  document.getElementById('info').textContent =
    j.params.toLocaleString() + ' parameters · device: ' + j.device +
    ' · checkpoint step ' + j.step + ' · 100% local, no external AI API';
});

document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = box.value.trim();
  if (!text) return;
  box.value = '';
  add('user', text);
  send.disabled = true; box.disabled = true;
  const thinking = add('ai', '…');
  try {
    const r = await fetch('chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    const j = await r.json();
    thinking.textContent = j.reply || '(empty reply)';
    if (j.system) thinking.className = 'msg sys';
  } catch (err) {
    thinking.textContent = 'error: ' + err;
  }
  send.disabled = false; box.disabled = false; box.focus();
});
</script>
</body>
</html>"""


class ChatState:
    def __init__(self, checkpoint: str, tokenizer_path: str):
        self.device = pick_device()
        self.model, self.ckpt = load_checkpoint(checkpoint, device=self.device)
        self.model.eval()
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.memory = Memory()
        self.session = time.strftime("web-%Y%m%d-%H%M%S")
        self.history: list[dict] = []
        self.lock = threading.Lock()  # one generation at a time

    def reply(self, user_msg: str) -> tuple[str, bool]:
        """Returns (text, is_system_notice)."""
        if user_msg.startswith("/remember "):
            fact = user_msg[len("/remember "):].strip()
            self.memory.add_fact(fact)
            return (f"Stored fact: \u201c{fact}\u201d (in a JSON file — NOT in the model's weights).", True)
        if user_msg == "/facts":
            facts = self.memory.get_facts()
            return (("Stored facts:\n- " + "\n- ".join(facts)) if facts else "No facts stored yet.", True)
        if user_msg == "/reset":
            self.history = []
            return ("Conversation context cleared.", True)

        with self.lock:
            ids = build_prompt_ids(self.tokenizer, self.memory.get_facts(),
                                   self.history, user_msg, self.model.cfg.context_length)
            out = generate(self.model, ids, max_new_tokens=200, temperature=0.7,
                           top_k=40, top_p=0.95, repetition_penalty=1.1,
                           stop_tokens={EOS, USER_TOK, ASSISTANT_TOK},
                           device=self.device)
            text = self.tokenizer.decode(out).strip() or "(the model produced only a stop token)"
            self.history.append({"role": "user", "text": user_msg})
            self.history.append({"role": "assistant", "text": text})
            self.memory.append_turn(self.session, "user", user_msg)
            self.memory.append_turn(self.session, "assistant", text)
            return (text, False)


STATE: ChatState | None = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
        elif self.path == "/info":
            info = {"params": STATE.model.num_parameters(), "device": STATE.device,
                    "step": STATE.ckpt.get("step")}
            self._send(200, json.dumps(info).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/chat":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            msg = json.loads(self.rfile.read(length))["message"]
            text, system = STATE.reply(str(msg)[:2000])
            self._send(200, json.dumps({"reply": text, "system": system}).encode(),
                       "application/json")
        except Exception as e:  # keep the demo server resilient
            self._send(500, json.dumps({"reply": f"server error: {e}"}).encode(),
                       "application/json")

    def log_message(self, *args):  # quiet
        pass


def main() -> None:
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="my_ai/checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="my_ai/checkpoints/tokenizer.json")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    print("loading model …")
    STATE = ChatState(args.checkpoint, args.tokenizer)
    print(f"model ready: {STATE.model.num_parameters():,} params on {STATE.device}")
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"chat UI at http://localhost:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
