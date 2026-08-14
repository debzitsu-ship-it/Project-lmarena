"""
Generate a small synthetic chat-format dataset from hand-written templates.

TRANSPARENCY: this text is produced by plain string templates written by us
below -- no AI API, no pretrained model. It exists only to teach our model
the *format* of a conversation (<|user|> question <eos> <|assistant|> answer
<eos>) and a handful of toy facts. A tiny model trained on this will parrot
these patterns; that is expected, and is honestly all it can do at this size.

Run: .venv/bin/python -m my_ai.data.make_chat_data
Writes: my_ai/data/raw/chat_synthetic.jsonl
"""
from __future__ import annotations

import json
import os
import random

TOPICS = {
    "trains": "Trains are long vehicles that run on steel rails. A locomotive pulls carriages full of people or goods. Trains are fast and can carry many passengers at once.",
    "cats": "Cats are small furry animals that people keep as pets. They purr when happy, hunt mice, and sleep for most of the day.",
    "dogs": "Dogs are loyal animals that live with people. They bark, wag their tails, and love to play fetch and go for walks.",
    "rain": "Rain is water that falls from clouds in small drops. It waters plants and fills rivers and lakes.",
    "the sun": "The sun is a huge ball of hot gas at the centre of our solar system. It gives Earth light and warmth.",
    "the moon": "The moon is a rocky ball that circles the Earth. It shines at night by reflecting sunlight.",
    "books": "Books are pages of written words bound together. People read books to learn things and to enjoy stories.",
    "rivers": "Rivers are long streams of water that flow across the land toward the sea. Fish live in rivers and boats travel on them.",
    "bread": "Bread is a food made from flour, water and yeast. The dough is baked in an oven until it is soft inside and crisp outside.",
    "music": "Music is organised sound made with instruments or voices. People sing and play music to express feelings and to celebrate.",
    "birds": "Birds are animals with feathers and wings. Most birds can fly, and many sing songs in the morning.",
    "snow": "Snow is frozen water that falls from clouds as soft white flakes. It covers the ground in winter and children play in it.",
    "trees": "Trees are tall plants with a wooden trunk and many leaves. They give shade, clean the air, and are home to birds.",
    "boats": "Boats are vehicles that float on water. Sails, oars or engines push them along rivers, lakes and seas.",
    "clocks": "Clocks are machines that measure time. Their hands or numbers show the hours and minutes of the day.",
}

QUESTION_FORMS = [
    "Explain {t}.", "What are {t}?", "Tell me about {t}.",
    "What is {t}?", "Describe {t}.", "Can you explain {t}?",
    "I want to know about {t}.", "Please tell me about {t}.",
]

GREETINGS = [
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi there.", "Hi! What would you like to talk about?"),
    ("Good morning.", "Good morning! I hope you are having a nice day."),
    ("How are you?", "I am a small language model, but I am working well. How can I help?"),
    ("Thank you.", "You are welcome! Ask me anything else you like."),
    ("Goodbye.", "Goodbye! It was nice talking with you."),
    ("What can you do?", "I can chat with you and answer simple questions about everyday things."),
    ("Who are you?", "I am a small AI assistant trained from scratch on a little dataset."),
]


def main(out_path: str = "my_ai/data/raw/chat_synthetic.jsonl",
         n_samples: int = 1200, seed: int = 7) -> None:
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for _ in range(n_samples):
        if rng.random() < 0.25:
            q, a = rng.choice(GREETINGS)
        else:
            topic, answer = rng.choice(list(TOPICS.items()))
            q = rng.choice(QUESTION_FORMS).format(t=topic)
            a = answer
        text = f"<|user|>{q}<eos><|assistant|>{a}<eos>"
        rows.append({"text": text})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} chat samples to {out_path}")


if __name__ == "__main__":
    main()
