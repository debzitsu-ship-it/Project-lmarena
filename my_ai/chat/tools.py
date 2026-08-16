"""
Agent tools: app-level abilities that work ALONGSIDE the neural model.

This is how real assistants do hard math: the app detects a calculation
and computes it exactly in code, instead of hoping the language model's
learned patterns get the digits right. The model handles language; tools
handle precision. Fully transparent: replies produced by a tool are
labelled so you always know what answered you.

Tool 1: calculator — safe arithmetic evaluator (+ - * / ^ % parentheses).
No eval(); a tiny recursive-descent parser, so nothing unsafe can run.
"""
from __future__ import annotations

import re


class _Parser:
    """Recursive-descent parser for arithmetic expressions."""

    def __init__(self, text: str):
        self.toks = re.findall(r"\d+\.?\d*|[()+\-*/^%]", text)
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> str | None:
        t = self.peek()
        self.i += 1
        return t

    def parse(self) -> float:
        v = self.expr()
        if self.peek() is not None:
            raise ValueError("trailing tokens")
        return v

    def expr(self) -> float:            # + -
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.next()
            r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def term(self) -> float:            # * / %
        v = self.power()
        while self.peek() in ("*", "/", "%"):
            op = self.next()
            r = self.power()
            if op == "*":
                v *= r
            elif op == "/":
                if r == 0:
                    raise ZeroDivisionError
                v /= r
            else:
                v %= r
        return v

    def power(self) -> float:           # ^
        v = self.atom()
        if self.peek() == "^":
            self.next()
            v = v ** self.power()
        return v

    def atom(self) -> float:
        t = self.next()
        if t == "(":
            v = self.expr()
            if self.next() != ")":
                raise ValueError("missing )")
            return v
        if t == "-":
            return -self.atom()
        if t is None or t in "()+*/^%":
            raise ValueError("bad token")
        return float(t)


_WORDS = [
    (r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\btimes\b", "*"),
    (r"\bmultiplied by\b", "*"), (r"\bdivided by\b", "/"), (r"\bx\b", "*"),
    (r"\bto the power of\b", "^"), (r"\bsquared\b", "^2"), (r"\bcubed\b", "^3"),
    (r"\bpercent of\b", "% of"),
]
_TRIGGER = re.compile(
    r"(what\s+is|whats|calculate|compute|how\s+much\s+is|solve|evaluate)?[\s:]*"
    r"([\d\s\.\+\-\*/\^%\(\)]+|\d.*(plus|minus|times|divided by|x|squared|cubed).*)",
    re.IGNORECASE)


def try_calculator(text: str) -> str | None:
    """If text looks like an arithmetic question, return the exact answer."""
    t = text.lower().strip().rstrip("?.!")
    for pat, rep in _WORDS:
        t = re.sub(pat, rep, t)
    # "25% of 80"
    m = re.search(r"([\d.]+)\s*%\s*of\s*([\d.]+)", t)
    if m:
        val = float(m.group(1)) / 100 * float(m.group(2))
        return _fmt(f"{m.group(1)}% of {m.group(2)}", val)
    # strip question words, keep only the math-y tail
    t = re.sub(r"^(what\s+is|whats|calculate|compute|how\s+much\s+is|solve|evaluate)\b[\s:]*", "", t)
    t = t.strip()
    # must contain a digit and an operator, and nothing but math afterwards
    if not re.fullmatch(r"[\d\s\.\+\-\*/\^%\(\)]+", t):
        return None
    if not (re.search(r"\d", t) and re.search(r"[\+\-\*/\^%]", t)):
        return None
    try:
        val = _Parser(t).parse()
    except Exception:
        return None
    return _fmt(t, val)


def _fmt(expr: str, val: float) -> str:
    shown = int(val) if float(val).is_integer() and abs(val) < 1e15 else round(val, 6)
    return f"{expr.strip()} = {shown}   [exact: computed by calculator tool, not the neural net]"
