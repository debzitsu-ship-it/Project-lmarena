"""
Generate a synthetic chat-format dataset from hand-written templates.

TRANSPARENCY: every sentence below was written by hand -- no AI API, no
pretrained model. Topics now include COMPUTING/PROGRAMMING vocabulary
(per the owner's goal), each with several fact-sentences recombined in
random order/length so the model learns to compose, not parrot.

Also includes:
- "I don't know" examples: questions outside the model's tiny world get
  honest deflections. This teaches the single most valuable behavior a
  small model can have -- admitting ignorance instead of confidently
  answering "what is a computer?" with bread facts.
- Multi-turn conversations so context carries across turns.

Run: .venv/bin/python -m my_ai.data.make_chat_data
Writes: my_ai/data/raw/chat_synthetic.jsonl
"""
from __future__ import annotations

import json
import os
import random

TOPIC_FACTS = {
    # ---------- everyday topics ----------
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
    ],
    "the moon": [
        "The moon is a rocky ball that circles the Earth.",
        "The moon shines at night by reflecting light from the sun.",
        "The moon changes shape in the sky through the month.",
        "The moon pulls the sea and makes the tides.",
    ],
    "books": [
        "Books are pages of written words bound together.",
        "People read books to learn new things and enjoy stories.",
        "A library is a building full of books anyone can borrow.",
        "Some books tell true facts and others tell made-up tales.",
    ],
    "rivers": [
        "Rivers are long streams of water that flow across the land.",
        "A river begins in the hills and flows toward the sea.",
        "Fish live in rivers and boats travel along them.",
        "People build towns beside rivers to use the water.",
    ],
    "music": [
        "Music is organised sound made with instruments or voices.",
        "People sing and play music to share their feelings.",
        "A song has a tune and often words to sing along.",
        "Music can make people feel happy, calm or excited.",
    ],
    "birds": [
        "Birds are animals with feathers, wings and beaks.",
        "Most birds can fly and many sing songs in the morning.",
        "Birds build nests to lay their eggs in.",
        "Some birds fly far away when winter comes.",
    ],
    "trees": [
        "Trees are tall plants with a wooden trunk and many branches.",
        "Trees give shade in summer and clean the air we breathe.",
        "Birds and squirrels make their homes in trees.",
        "Fruit like apples and mangoes grow on trees.",
    ],
    # ---------- computing & programming topics ----------
    "computers": [
        "A computer is a machine that follows instructions to work with information.",
        "Computers have a processor that does the thinking and memory that holds data.",
        "People use computers to write, calculate, play games and talk to others.",
        "A computer only does exactly what its instructions tell it to do.",
        "Laptops, desktops and even phones are all kinds of computers.",
    ],
    "programming": [
        "Programming means writing instructions that a computer can follow.",
        "A program is a list of steps the computer runs one by one.",
        "Programmers write code, test it, and fix it when it breaks.",
        "Good programs are written in small clear steps.",
        "Programming languages let people give orders to machines.",
    ],
    "python": [
        "Python is a popular programming language that is easy to read.",
        "Python code uses simple words and spacing instead of heavy symbols.",
        "People use Python for websites, data science and artificial intelligence.",
        "A Python program runs line by line from top to bottom.",
        "Python was named after a comedy group, not the snake.",
    ],
    "a variable": [
        "A variable is a named box in a program that stores a value.",
        "You can put a number or some text into a variable and use it later.",
        "The value inside a variable can change while the program runs.",
        "Variables let programs remember things between steps.",
    ],
    "a function": [
        "A function is a named block of code that does one job.",
        "You call a function when you want its job done.",
        "Functions can take inputs and give back an output.",
        "Splitting code into functions keeps programs tidy and reusable.",
    ],
    "a loop": [
        "A loop repeats the same instructions many times.",
        "Loops save programmers from writing the same line again and again.",
        "A loop can run a fixed number of times or until something changes.",
        "Forgetting to stop a loop makes the program run forever.",
    ],
    "a bug": [
        "A bug is a mistake in a program that makes it behave wrongly.",
        "Programmers find bugs by testing and reading their code carefully.",
        "Fixing bugs is called debugging.",
        "Even tiny typing mistakes can cause big bugs.",
    ],
    "the internet": [
        "The internet is a huge network of computers connected around the world.",
        "Computers on the internet send information to each other in small packets.",
        "Websites, email and video calls all travel over the internet.",
        "A web browser is a program used to visit websites.",
    ],
    "artificial intelligence": [
        "Artificial intelligence means computer programs that learn patterns from data.",
        "An AI language model learns to guess the next word from many examples.",
        "AI does not think like a person; it follows mathematics.",
        "The more good data an AI is trained on, the better its guesses become.",
        "I myself am a very small AI model trained from scratch.",
    ],
    "a file": [
        "A file is a package of information saved on a computer.",
        "Files have names and endings that show what kind they are.",
        "Documents, pictures and programs are all stored as files.",
        "Folders keep files organised so people can find them.",
    ],
    "a keyboard": [
        "A keyboard is the set of buttons used to type into a computer.",
        "Each key sends a letter, number or command to the computer.",
        "People press keys to write words and control programs.",
    ],
    "memory": [
        "Computer memory is where a computer keeps information while it works.",
        "Fast memory called RAM is emptied when the power goes off.",
        "Storage like a disk keeps files even when the computer is off.",
        "More memory lets a computer work on bigger jobs at once.",
    ],
}

QUESTION_FORMS = [
    "Explain {t}.", "What are {t}?", "Tell me about {t}.", "What is {t}?",
    "Describe {t}.", "Can you explain {t}?", "I want to know about {t}.",
    "Please tell me about {t}.", "What do you know about {t}?",
    "Say something about {t}.", "Teach me about {t}.", "How would you describe {t}?",
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
    ("What can you do?", ["I can chat about everyday things and simple computer topics.", "I can answer simple questions about animals, weather, books and computers."]),
    ("Who are you?", ["I am a small AI assistant trained from scratch on a little dataset.", "I am a tiny language model that was trained from random weights."]),
    ("Who made you?", ["I was built and trained from scratch by my owner, using their own code and data."]),
]

# Tiny code-writing requests with hand-written correct answers. At this
# model scale these teach the *shape* of code answers; real code fluency
# comes from the raw source-code corpus + more parameters later.
CODE_QA = [
    ("Write a Python program that prints hello.",
     'print("hello")'),
    ("How do I print something in Python?",
     'Use the print function, like this: print("your text")'),
    ("Write a Python function that adds two numbers.",
     "def add(a, b):\n    return a + b"),
    ("Show me a Python loop that counts to five.",
     "for i in range(1, 6):\n    print(i)"),
    ("How do I make a variable in Python?",
     "Just give it a name and a value, like this: x = 10"),
    ("Write a function that doubles a number.",
     "def double(n):\n    return n * 2"),
    ("How do I write a comment in Python?",
     "Start the line with a hash sign, like this: # this is a comment"),
    ("Show me an if statement in Python.",
     'if x > 5:\n    print("big")\nelse:\n    print("small")'),
    ("Write a Python function that says hello to a name.",
     'def greet(name):\n    print("Hello, " + name)'),
    ("How do I make a list in Python?",
     "Use square brackets, like this: fruits = [\"apple\", \"mango\", \"banana\"]"),
    ("Show me how to loop over a list in Python.",
     'for fruit in fruits:\n    print(fruit)'),
    ("Write a function that checks if a number is even.",
     "def is_even(n):\n    return n % 2 == 0"),
]

# Questions the model should honestly deflect. The exact questions vary but
# the ANSWER PATTERN ("I am not sure...") is what it must learn.
UNKNOWN_QUESTIONS = [
    "What is quantum physics?", "Who is the president?", "What is the capital of France?",
    "How do airplanes fly?", "What is the meaning of life?", "Explain chemistry.",
    "What is calculus?", "Tell me about history.", "Who won the world cup?",
    "What is the stock market?", "Explain medicine.", "What is philosophy?",
    "How do rockets work?", "What is electricity?", "Tell me about space travel.",
    "What is biology?", "Who invented the telephone?", "What is economics?",
]
UNKNOWN_ANSWERS = [
    "I am not sure about that. I am a small model and I only know simple everyday and computer topics.",
    "I do not know enough to answer that well. Try asking me about animals, weather, books or computers.",
    "That is beyond what I have learned so far. I know simple things like cats, trains, rain and programming.",
    "I cannot answer that properly yet. My training data is small, so I only know basic topics.",
]


def build_answer(rng: random.Random, facts: list[str]) -> str:
    n = rng.choice([2, 2, 3])
    return " ".join(rng.sample(facts, min(n, len(facts))))


def main(out_path: str = "my_ai/data/raw/chat_synthetic.jsonl",
         n_samples: int = 3000, seed: int = 7) -> None:
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    topics = list(TOPIC_FACTS.items())
    for _ in range(n_samples):
        r = rng.random()
        if r < 0.15:                                   # greetings
            q, answers = rng.choice(GREETINGS)
            text = f"<|user|>{q}<eos><|assistant|>{rng.choice(answers)}<eos>"
        elif r < 0.30:                                 # honest "I don't know"
            q = rng.choice(UNKNOWN_QUESTIONS)
            a = rng.choice(UNKNOWN_ANSWERS)
            text = f"<|user|>{q}<eos><|assistant|>{a}<eos>"
        elif r < 0.45:                                 # code-writing Q&A
            q, a = rng.choice(CODE_QA)
            text = f"<|user|>{q}<eos><|assistant|>{a}<eos>"
        elif r < 0.85:                                 # topic Q&A
            topic, facts = rng.choice(topics)
            q = rng.choice(QUESTION_FORMS).format(t=topic)
            text = f"<|user|>{q}<eos><|assistant|>{build_answer(rng, facts)}<eos>"
        else:                                          # multi-turn
            g, ganswers = rng.choice(GREETINGS[:4])
            t1, f1 = rng.choice(topics)
            t2, f2 = rng.choice(topics)
            q1 = rng.choice(QUESTION_FORMS).format(t=t1)
            q2 = rng.choice(QUESTION_FORMS).format(t=t2)
            text = (f"<|user|>{g}<eos><|assistant|>{rng.choice(ganswers)}<eos>"
                    f"<|user|>{q1}<eos><|assistant|>{build_answer(rng, f1)}<eos>"
                    f"<|user|>{q2}<eos><|assistant|>{build_answer(rng, f2)}<eos>")
        rows.append({"text": text})
    with open(out_path, "w", encoding="utf-8") as f:
        for r_ in rows:
            f.write(json.dumps(r_, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} chat samples to {out_path} "
          f"({len(TOPIC_FACTS)} topics incl. computing, {len(UNKNOWN_QUESTIONS)} IDK questions)")


if __name__ == "__main__":
    main()
