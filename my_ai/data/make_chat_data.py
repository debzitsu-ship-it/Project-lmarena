"""
Generate a synthetic chat-format dataset from hand-written templates.

TRANSPARENCY: all text below was written by hand -- no AI API, no pretrained
model. Each topic now has SEVERAL answer variants and sentence fragments that
get recombined, so the model sees varied phrasings of the same facts. That
pushes it toward recombining language instead of parroting one fixed string.

Run: .venv/bin/python -m my_ai.data.make_chat_data
Writes: my_ai/data/raw/chat_synthetic.jsonl
"""
from __future__ import annotations

import json
import os
import random

# Each topic: list of fact-sentences (hand-written). Answers are built by
# sampling 2-3 of them in random order, so wording varies between examples.
TOPIC_FACTS = {
    "trains": [
        "Trains are long vehicles that run on steel rails.",
        "A locomotive pulls the carriages along the track.",
        "Trains carry many passengers and heavy goods at once.",
        "Trains travel between stations and can be very fast.",
        "People ride trains to work and to visit other cities.",
    ],
    "cats": [
        "Cats are small furry animals that people keep as pets.",
        "A cat purrs when it is happy and content.",
        "Cats hunt mice and love to chase small things.",
        "Cats sleep for most of the day, often in warm places.",
        "A cat washes itself with its tongue and paws.",
    ],
    "dogs": [
        "Dogs are loyal animals that live with people.",
        "A dog barks to warn its family and wags its tail when happy.",
        "Dogs love to play fetch and go for long walks.",
        "Many dogs can learn tricks and simple commands.",
        "A dog is often called a person's best friend.",
    ],
    "rain": [
        "Rain is water that falls from clouds in small drops.",
        "Rain waters the plants and fills the rivers and lakes.",
        "After heavy rain the streets are wet and shiny.",
        "Farmers need rain so their crops can grow.",
        "Clouds turn grey and heavy just before it rains.",
    ],
    "the sun": [
        "The sun is a huge ball of hot gas at the centre of our solar system.",
        "The sun gives the Earth light and warmth every day.",
        "Plants use sunlight to grow and make food.",
        "The sun rises in the morning and sets in the evening.",
        "Without the sun the Earth would be dark and frozen.",
    ],
    "the moon": [
        "The moon is a rocky ball that circles the Earth.",
        "The moon shines at night by reflecting light from the sun.",
        "The moon changes shape in the sky through the month.",
        "Astronauts have walked on the moon and brought back rocks.",
        "The moon pulls the sea and makes the tides.",
    ],
    "books": [
        "Books are pages of written words bound together.",
        "People read books to learn new things and enjoy stories.",
        "A library is a building full of books anyone can borrow.",
        "Some books tell true facts and others tell made-up tales.",
        "Reading a good book can feel like visiting another world.",
    ],
    "rivers": [
        "Rivers are long streams of water that flow across the land.",
        "A river begins in the hills and flows toward the sea.",
        "Fish live in rivers and boats travel along them.",
        "People build towns beside rivers to use the water.",
        "A small river is called a stream or a brook.",
    ],
    "bread": [
        "Bread is a food made from flour, water and yeast.",
        "The dough is baked in a hot oven until it is golden.",
        "Fresh bread is soft inside with a crisp crust outside.",
        "People eat bread with butter, cheese or jam.",
        "Bakers wake up early to bake bread for the day.",
    ],
    "music": [
        "Music is organised sound made with instruments or voices.",
        "People sing and play music to share their feelings.",
        "A song has a tune and often words to sing along.",
        "Drums keep the beat while other instruments play the melody.",
        "Music can make people feel happy, calm or excited.",
    ],
    "birds": [
        "Birds are animals with feathers, wings and beaks.",
        "Most birds can fly and many sing songs in the morning.",
        "Birds build nests to lay their eggs in.",
        "Some birds fly far away when winter comes.",
        "A baby bird hatches from an egg and is fed by its parents.",
    ],
    "snow": [
        "Snow is frozen water that falls as soft white flakes.",
        "Snow covers the ground in winter like a white blanket.",
        "Children build snowmen and throw snowballs in the snow.",
        "Every snowflake has its own special shape.",
        "When the sun comes out the snow melts into water.",
    ],
    "trees": [
        "Trees are tall plants with a wooden trunk and many branches.",
        "Trees give shade in summer and clean the air we breathe.",
        "Birds and squirrels make their homes in trees.",
        "In autumn the leaves of many trees turn red and gold.",
        "Fruit like apples and mangoes grow on trees.",
    ],
    "boats": [
        "Boats are vehicles that float and travel on water.",
        "Sails, oars or engines push boats along.",
        "Fishermen use boats to catch fish in rivers and seas.",
        "Big ships carry goods across the ocean.",
        "A small boat can be rowed with two oars.",
    ],
    "clocks": [
        "Clocks are machines that measure and show the time.",
        "The hands of a clock point to the hours and minutes.",
        "An alarm clock rings to wake people in the morning.",
        "Some clocks tick and some are silent and digital.",
        "People look at clocks so they are not late.",
    ],
}

QUESTION_FORMS = [
    "Explain {t}.", "What are {t}?", "Tell me about {t}.", "What is {t}?",
    "Describe {t}.", "Can you explain {t}?", "I want to know about {t}.",
    "Please tell me about {t}.", "What do you know about {t}?",
    "Say something about {t}.", "Teach me about {t}.",
]

GREETINGS = [
    ("Hello!", ["Hello! How can I help you today?", "Hello! What would you like to know?", "Hello there! Ask me anything you like."]),
    ("Hi there.", ["Hi! What would you like to talk about?", "Hi there! How can I help?"]),
    ("Good morning.", ["Good morning! I hope you are having a nice day.", "Good morning! What shall we talk about?"]),
    ("How are you?", ["I am a small language model, but I am working well. How can I help?", "I am doing fine, thank you. What would you like to know?"]),
    ("Thank you.", ["You are welcome! Ask me anything else you like.", "You are welcome! I am happy to help."]),
    ("Thanks!", ["No problem! Is there anything else you want to know?"]),
    ("Goodbye.", ["Goodbye! It was nice talking with you.", "Goodbye! Come back any time."]),
    ("Bye.", ["Bye! Have a lovely day."]),
    ("What can you do?", ["I can chat with you and answer simple questions about everyday things.", "I can talk about simple topics like animals, weather and books."]),
    ("Who are you?", ["I am a small AI assistant trained from scratch on a little dataset.", "I am a tiny language model that was trained from random weights."]),
    ("Who made you?", ["I was built and trained from scratch by my owner, using their own code and data."]),
]


def build_answer(rng: random.Random, facts: list[str]) -> str:
    n = rng.choice([2, 2, 3])
    return " ".join(rng.sample(facts, min(n, len(facts))))


def main(out_path: str = "my_ai/data/raw/chat_synthetic.jsonl",
         n_samples: int = 2000, seed: int = 7) -> None:
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    topics = list(TOPIC_FACTS.items())
    for _ in range(n_samples):
        r = rng.random()
        if r < 0.2:
            q, answers = rng.choice(GREETINGS)
            text = f"<|user|>{q}<eos><|assistant|>{rng.choice(answers)}<eos>"
        elif r < 0.9:
            topic, facts = rng.choice(topics)
            q = rng.choice(QUESTION_FORMS).format(t=topic)
            text = f"<|user|>{q}<eos><|assistant|>{build_answer(rng, facts)}<eos>"
        else:
            # two-turn conversation: greeting then a topic question
            g, ganswers = rng.choice(GREETINGS[:4])
            topic, facts = rng.choice(topics)
            q = rng.choice(QUESTION_FORMS).format(t=topic)
            text = (f"<|user|>{g}<eos><|assistant|>{rng.choice(ganswers)}<eos>"
                    f"<|user|>{q}<eos><|assistant|>{build_answer(rng, facts)}<eos>")
        rows.append({"text": text})
    with open(out_path, "w", encoding="utf-8") as f:
        for r_ in rows:
            f.write(json.dumps(r_, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} chat samples to {out_path}")


if __name__ == "__main__":
    main()
