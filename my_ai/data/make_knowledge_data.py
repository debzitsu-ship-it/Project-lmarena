"""
Generate a general-knowledge Q&A dataset from hand-written fact tables.
No AI involved: facts below are typed by hand; questions are template-varied.

Covers: world capitals, planets, continents/oceans, famous basics, units,
human body, animals records. Each fact appears under many phrasings so the
model learns the FACT, not one sentence.

Run: python -m my_ai.data.make_knowledge_data
Writes: my_ai/data/raw/knowledge_synthetic.jsonl
"""
from __future__ import annotations

import json
import os
import random

CAPITALS = {
    "France": "Paris", "India": "New Delhi", "Japan": "Tokyo", "Italy": "Rome",
    "Germany": "Berlin", "Spain": "Madrid", "Russia": "Moscow", "China": "Beijing",
    "England": "London", "the United States": "Washington, D.C.", "Egypt": "Cairo",
    "Brazil": "Brasilia", "Canada": "Ottawa", "Australia": "Canberra",
    "Greece": "Athens", "Portugal": "Lisbon", "Nepal": "Kathmandu",
    "Bangladesh": "Dhaka", "Pakistan": "Islamabad", "Sri Lanka": "Colombo",
}
CAP_Q = ["What is the capital of {k}?", "Capital of {k}?", "Tell me the capital of {k}.",
         "Which city is the capital of {k}?", "Name the capital of {k}."]

PLANETS = [
    ("Mercury", "Mercury is the closest planet to the sun and the smallest planet."),
    ("Venus", "Venus is the hottest planet, covered in thick clouds."),
    ("Earth", "Earth is our home planet, the only one known to have life."),
    ("Mars", "Mars is called the red planet because of its rusty dust."),
    ("Jupiter", "Jupiter is the largest planet, a giant ball of gas."),
    ("Saturn", "Saturn is famous for its beautiful rings of ice and rock."),
    ("Uranus", "Uranus is an ice giant that spins on its side."),
    ("Neptune", "Neptune is the farthest planet, a cold blue ice giant."),
]

FACTS = [
    ("How many continents are there?", "There are seven continents: Asia, Africa, North America, South America, Antarctica, Europe and Australia."),
    ("How many oceans are there?", "There are five oceans: the Pacific, Atlantic, Indian, Southern and Arctic."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth."),
    ("What is the longest river?", "The Nile in Africa is usually called the longest river on Earth."),
    ("What is the highest mountain?", "Mount Everest is the highest mountain above sea level."),
    ("What is the largest animal?", "The blue whale is the largest animal that has ever lived."),
    ("What is the fastest land animal?", "The cheetah is the fastest land animal."),
    ("What is the tallest animal?", "The giraffe is the tallest living animal."),
    ("How many days are in a year?", "A year has 365 days, and a leap year has 366."),
    ("How many hours are in a day?", "A day has 24 hours."),
    ("How many minutes are in an hour?", "An hour has 60 minutes."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("How many legs does an insect have?", "An insect has six legs."),
    ("What do bees make?", "Bees make honey and help plants by carrying pollen."),
    ("What gas do humans breathe in?", "Humans breathe in oxygen from the air."),
    ("What gas do plants take in?", "Plants take in carbon dioxide and give out oxygen."),
    ("How many bones are in the human body?", "An adult human body has 206 bones."),
    ("What pumps blood in the body?", "The heart pumps blood around the body."),
    ("What is water made of?", "Water is made of hydrogen and oxygen; its formula is H2O."),
    ("What is the closest star to Earth?", "The sun is the closest star to Earth."),
    ("How many planets are in the solar system?", "There are eight planets in the solar system."),
    ("What is the largest planet?", "Jupiter is the largest planet in the solar system."),
    ("What is the smallest planet?", "Mercury is the smallest planet in the solar system."),
    ("How many colours are in a rainbow?", "A rainbow has seven colours: red, orange, yellow, green, blue, indigo and violet."),
    ("How many sides does a triangle have?", "A triangle has three sides."),
    ("How many sides does a square have?", "A square has four equal sides."),
]
FACT_PREFIX = ["", "", "Tell me, ", "Please, ", "Quick question: ", "Hey, "]

PLANET_Q = ["Tell me about {p}.", "What is {p}?", "Describe the planet {p}.",
            "What do you know about {p}?"]


def main(out_path: str = "my_ai/data/raw/knowledge_synthetic.jsonl",
         n_samples: int = 3000, seed: int = 21) -> None:
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    caps = list(CAPITALS.items())
    for _ in range(n_samples):
        r = rng.random()
        if r < 0.35:
            k, v = rng.choice(caps)
            q = rng.choice(CAP_Q).format(k=k)
            a = f"The capital of {k} is {v}."
        elif r < 0.55:
            p, desc = rng.choice(PLANETS)
            q = rng.choice(PLANET_Q).format(p=p)
            a = desc
        else:
            q0, a = rng.choice(FACTS)
            q = rng.choice(FACT_PREFIX) + q0
        rows.append({"text": f"<|user|>{q}<eos><|assistant|>{a}<eos>"})
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} knowledge samples to {out_path}")


if __name__ == "__main__":
    main()
