"""Project-authored short questions with short answers, released as CC0-1.0.

Each fact appears in several phrasings so the model learns the fact rather than one string.
The items tested by data/simple_questions_holdout.json (4 plus 9, days in a week, the
opposite of tall, a home address, and so on) are left out on purpose, so a holdout gain
measures transfer to unseen questions instead of recall of trained ones.
"""

from __future__ import annotations

SOURCE = "project-authored/compute.short_facts"
LICENSE = "CC0-1.0"

NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()

# (a, b) pairs per operation that the holdout asks about, in both orders where it matters.
HOLDOUT_ARITHMETIC = {
    "plus": {(4, 9), (9, 4)},
    "minus": {(15, 6)},
    "times": {(3, 4), (4, 3)},
    "divided by": {(18, 3)},
}
HOLDOUT_COMPARISONS = {(8, 3), (3, 8)}

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")

# (one-word answer, sentence answer, phrasings). The one-word answer is written exactly as it
# appears in the sentence. Holdout facts (days per week, months per year, meow, hearing) are absent.
FACTS = (
    ("blue", "The sky is blue on a clear day.",
     ("What color is the sky on a clear day?", "On a sunny day, what color is the sky?",
      "What is the color of a clear daytime sky?")),
    ("green", "Grass is green.",
     ("What color is grass?", "What color is healthy grass?", "Grass is usually what color?")),
    ("yellow", "A ripe banana is yellow.",
     ("What color is a ripe banana?", "Ripe bananas are what color?")),
    ("white", "Snow is white.", ("What color is snow?", "Fresh snow is what color?")),
    ("red", "A ripe strawberry is red.",
     ("What color is a ripe strawberry?", "Ripe strawberries are what color?")),
    ("dog", "A dog barks.",
     ("Which animal barks?", "What animal says woof?", "Which pet barks at strangers?")),
    ("cow", "A cow says moo.", ("Which animal says moo?", "What farm animal says moo?")),
    ("duck", "A duck says quack.", ("Which animal says quack?", "What bird says quack?")),
    ("sheep", "A sheep says baa.", ("Which animal says baa?", "What farm animal says baa?")),
    ("Bees", "Bees make honey.", ("Which insect makes honey?", "What makes honey?")),
    ("fish", "A fish lives in water and has fins.",
     ("Which animal lives in water and has fins?", "What animal has gills and swims?")),
    ("eyes", "People see with their eyes.",
     ("What do people use to see?", "Which body part do we see with?",
      "What do you use to see: ears or eyes?")),
    ("nose", "People smell with their nose.",
     ("What do people use to smell?", "Which body part do we smell with?")),
    ("tongue", "People taste with their tongue.",
     ("What do people use to taste food?", "Which body part do we taste with?")),
    ("hands", "People usually write with their hands.",
     ("What do people usually write with: hands or feet?",)),
    ("sun", "The sun gives Earth light during the day.",
     ("What gives us light during the day?", "What bright star do we see in the daytime sky?")),
    ("moon", "The moon is often seen in the night sky.",
     ("What round object do we often see in the night sky?",
      "What orbits Earth and shines at night?")),
    ("water", "Fish live in water.", ("Where do fish live?", "Fish swim in what?")),
    ("ice", "Water turns into ice when it freezes.",
     ("What does water become when it freezes?", "Frozen water is called what?")),
    ("steam", "Boiling water turns into steam.",
     ("What does boiling water turn into?", "Water that boils becomes what?")),
    ("cold", "Ice is cold.", ("Is ice hot or cold?", "How does ice feel: hot or cold?")),
    ("hot", "Fire is hot.", ("Is fire hot or cold?", "How does fire feel: hot or cold?")),
    ("4", "A dog has 4 legs.", ("How many legs does a dog have?", "A dog walks on how many legs?")),
    ("2", "A bird has 2 legs.",
     ("How many legs does a bird have?", "A chicken stands on how many legs?")),
    ("8", "A spider has 8 legs.", ("How many legs does a spider have?",)),
    ("6", "An insect has 6 legs.", ("How many legs does an insect have?",)),
    ("5", "A hand has 5 fingers.",
     ("How many fingers are on one hand?", "Count the fingers on one hand. How many are there?")),
    ("10", "People have 10 fingers.", ("How many fingers do people have on both hands together?",)),
    ("3", "A triangle has 3 sides.",
     ("How many sides does a triangle have?", "A triangle has how many corners?")),
    ("4", "A square has 4 sides.", ("How many sides does a square have?",)),
    ("60", "An hour has 60 minutes.",
     ("How many minutes are in one hour?", "One hour equals how many minutes?")),
    ("60", "A minute has 60 seconds.",
     ("How many seconds are in one minute?", "One minute equals how many seconds?")),
    ("24", "A day has 24 hours.", ("How many hours are in one day?", "One day lasts how many hours?")),
    ("4", "A year has 4 seasons.", ("How many seasons are in a year?",)),
    ("Winter", "Winter is usually the coldest season.",
     ("Which season is usually the coldest?", "In which season does it often snow?")),
    ("Summer", "Summer is usually the hottest season.", ("Which season is usually the hottest?",)),
    ("Paris", "Paris is the capital of France.",
     ("What is the capital of France?", "Which city is the capital of France?")),
    ("London", "London is the capital of England.",
     ("What is the capital of England?", "Which city is the capital of England?")),
    ("Tokyo", "Tokyo is the capital of Japan.",
     ("What is the capital of Japan?", "Which city is the capital of Japan?")),
    ("Rome", "Rome is the capital of Italy.", ("What is the capital of Italy?",)),
    ("Madrid", "Madrid is the capital of Spain.", ("What is the capital of Spain?",)),
    ("Ottawa", "Ottawa is the capital of Canada.", ("What is the capital of Canada?",)),
    ("English", "People in England mainly speak English.",
     ("What language do most people in England speak?",)),
    ("Earth", "We live on the planet Earth.",
     ("What planet do we live on?", "What is the name of our planet?")),
    ("Mars", "Mars is called the red planet.", ("Which planet is called the red planet?",)),
    ("Pacific", "The Pacific Ocean is the largest ocean.",
     ("What is the largest ocean on Earth?", "Which ocean is the biggest?")),
    ("7", "There are 7 continents.", ("How many continents are there?",)),
    ("kitchen", "People usually cook in the kitchen.", ("In which room do people usually cook?",)),
    ("bed", "People usually sleep in a bed.", ("What do people usually sleep on at night?",)),
    ("umbrella", "An umbrella keeps you dry in the rain.",
     ("What do you hold over your head to stay dry in the rain?",)),
    ("key", "A key opens a lock.", ("What do you use to open a lock?",)),
    ("clock", "A clock tells the time.", ("What do you look at to tell the time?",)),
    ("doctor", "A doctor helps sick people get better.",
     ("Who helps sick people get better?", "Which person do you visit when you are ill?")),
    ("teacher", "A teacher teaches students at school.", ("Who teaches students at school?",)),
    ("farmer", "A farmer grows crops.", ("Who grows crops on a farm?",)),
    ("apple", "An apple is a fruit.", ("Which is a fruit: an apple or a carrot?",)),
    ("carrot", "A carrot is a vegetable.", ("Which is a vegetable: a carrot or a banana?",)),
    ("milk", "Cows give milk.", ("What drink do cows give?",)),
    ("wings", "Birds fly with their wings.",
     ("What do birds use to fly?", "Which body parts help a bird fly?")),
    ("night", "The sky is dark at night.", ("Is the sky dark during the day or at night?",)),
    ("east", "The sun rises in the east.",
     ("In which direction does the sun rise?", "Does the sun come up in the east or the west?")),
    ("west", "The sun sets in the west.", ("In which direction does the sun set?",)),
    ("30", "30 is ten more than 20.", ("What number is ten more than twenty?",)),
    ("10", "10 tens make 100.", ("How many tens make one hundred?",)),
)

# Each word appears in one pair only, so every "opposite of X" prompt has one answer.
# "tall" and "short" are holdout items, so "long" is absent too.
OPPOSITES = (
    ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("happy", "sad"), ("up", "down"),
    ("open", "closed"), ("light", "dark"), ("day", "night"), ("old", "young"), ("wet", "dry"),
    ("full", "empty"), ("hard", "soft"), ("early", "late"), ("high", "low"), ("strong", "weak"),
    ("loud", "quiet"), ("rich", "poor"), ("clean", "dirty"), ("left", "right"), ("in", "out"),
    ("thick", "thin"), ("near", "far"), ("push", "pull"), ("win", "lose"), ("buy", "sell"),
    ("first", "last"), ("before", "after"), ("above", "below"), ("start", "finish"),
    ("true", "false"), ("yes", "no"), ("good", "bad"), ("inside", "outside"),
    ("asleep", "awake"), ("always", "never"), ("sweet", "sour"), ("wide", "narrow"),
    ("deep", "shallow"), ("give", "take"), ("arrive", "leave"), ("laugh", "cry"),
    ("remember", "forget"), ("friend", "enemy"), ("tall", "short"),
)
HOLDOUT_OPPOSITES = {"tall", "short"}

PLURALS = (
    ("cat", "cats"), ("dog", "dogs"), ("car", "cars"), ("tree", "trees"), ("pen", "pens"),
    ("apple", "apples"), ("house", "houses"), ("bird", "birds"), ("chair", "chairs"),
    ("table", "tables"), ("flower", "flowers"), ("girl", "girls"), ("boy", "boys"),
    ("box", "boxes"), ("bus", "buses"), ("dish", "dishes"), ("glass", "glasses"),
    ("fox", "foxes"), ("watch", "watches"), ("brush", "brushes"), ("baby", "babies"),
    ("city", "cities"), ("story", "stories"), ("lady", "ladies"), ("party", "parties"),
    ("leaf", "leaves"), ("knife", "knives"), ("wolf", "wolves"), ("life", "lives"),
    ("man", "men"), ("woman", "women"), ("child", "children"), ("foot", "feet"),
    ("tooth", "teeth"), ("mouse", "mice"), ("goose", "geese"), ("person", "people"),
    ("sheep", "sheep"), ("fish", "fish"), ("deer", "deer"), ("potato", "potatoes"),
    ("tomato", "tomatoes"), ("hero", "heroes"), ("toy", "toys"), ("key", "keys"),
    ("day", "days"), ("cup", "cups"), ("shoe", "shoes"), ("song", "songs"), ("egg", "eggs"),
    ("book", "books"),
)
HOLDOUT_PLURALS = {"book"}

COPY_WORDS = ("apple", "river", "garden", "yellow", "pencil", "cloud", "music", "bridge",
              "orange", "ladder", "planet", "candle", "forest", "rabbit", "mirror", "bottle",
              "blanket", "tiger", "lemon", "castle", "window")
HOLDOUT_COPY = {"window"}

# (phrasings, answer). A home address and a number the user is thinking of are holdout items.
REFUSALS = (
    (("What is my name?", "Do you know my name?", "Tell me my name."),
     "I don't know your name. You haven't told me."),
    (("How old am I?", "What is my age?", "Can you tell me how old I am?"),
     "I don't know your age. You haven't told me."),
    (("What is my favorite color?", "Which color do I like best?"),
     "I don't know your favorite color. You haven't told me."),
    (("What is my phone number?", "Tell me my phone number."),
     "I don't know your phone number. You haven't told me."),
    (("What did I eat for breakfast today?", "What was my breakfast this morning?"),
     "I don't know what you ate. You haven't told me."),
    (("What is my dog's name?", "What is my pet called?"),
     "I don't know your pet's name. You haven't told me."),
    (("When is my birthday?", "What day is my birthday?"),
     "I don't know your birthday. You haven't told me."),
    (("Which city do I live in?", "Which country am I in right now?"),
     "I don't know where you are. You haven't told me."),
    (("What is in my pocket right now?", "What am I holding in my hand?"),
     "I can't see you, so I don't know."),
    (("What am I wearing today?", "What color is my shirt?"),
     "I can't see you, so I don't know what you are wearing."),
    (("What is my best friend's name?", "Who is my best friend?"),
     "I don't know who your best friend is. You haven't told me."),
    (("What word did I write on the paper?", "What did I just draw?"),
     "I don't know. You haven't shown me."),
    (("What is my password?", "Tell me the password to my account."),
     "I don't know your password, and you should not share it."),
    (("What will I dream about tonight?", "What will I get for my next birthday?"),
     "I don't know. Nobody can know that for sure."),
    (("What is my teacher's name?", "Who is my teacher?"),
     "I don't know your teacher's name. You haven't told me."),
    (("How many brothers do I have?", "How many sisters do I have?"),
     "I don't know. You haven't told me about your family."),
    (("What time did I wake up today?", "When did I go to bed last night?"),
     "I don't know. You haven't told me."),
    (("Which card am I holding?", "What letter am I thinking of?"),
     "I don't know. You haven't given me any clues."),
)
REFUSAL_WRAPPERS = ("{q}", "Quick question: {q}", "{q} Answer briefly.")
WORD_WRAPPERS = ("Please answer: ", "Question: ")

ARITHMETIC_TEMPLATES = {
    "plus": ("What is {a} plus {b}? Reply with the number.", "What is {a} + {b}?",
             "Add {a} and {b}.", "{a} plus {b} equals what?"),
    "minus": ("What is {a} minus {b}? Reply with the number.", "What is {a} - {b}?",
              "Subtract {b} from {a}.", "{a} take away {b} leaves how many?"),
    "times": ("What is {a} times {b}? Reply with the number.", "What is {a} x {b}?",
              "Multiply {a} by {b}.", "What do you get when you multiply {a} and {b}?"),
    "divided by": ("What is {a} divided by {b}? Reply with the number.", "What is {a} / {b}?",
                   "Divide {a} by {b}.", "How many times does {b} go into {a}?"),
}


def _row(prompt: str, answer: str, category: str, group: str) -> dict:
    # group ties every phrasing of one fact together so a split never separates them.
    return {"prompt": prompt, "answer": answer, "category": category, "group": group}


def arithmetic_rows() -> list[dict]:
    cases = []
    for a in range(13):
        for b in range(13):
            cases.append(("plus", a, b, a + b))
            cases.append(("minus", a + b, b, a))
    for a in range(1, 11):
        for b in range(1, 11):
            cases.append(("times", a, b, a * b))
            cases.append(("divided by", a * b, b, a))
    rows = []
    for operation, a, b, result in cases:
        if (a, b) in HOLDOUT_ARITHMETIC[operation]:
            continue
        for template in ARITHMETIC_TEMPLATES[operation]:
            rows.append(_row(template.replace("{a}", str(a)).replace("{b}", str(b)), str(result),
                             "math", f"arithmetic:{operation}:{a}:{b}"))
    return rows


def comparison_rows() -> list[dict]:
    rows = []
    for a in range(13):
        for b in range(13):
            if a == b or (a, b) in HOLDOUT_COMPARISONS:
                continue
            group, answer = f"compare:{a}:{b}", "yes" if a > b else "no"
            rows += [
                _row(f"Answer yes or no: Is {NUMBER_WORDS[a]} greater than {NUMBER_WORDS[b]}?",
                     answer, "instruction", group),
                _row(f"Is {a} bigger than {b}? Answer yes or no.", answer, "instruction", group),
                _row(f"Which is larger, {a} or {b}?", str(max(a, b)), "math", group),
            ]
    return rows


def calendar_rows() -> list[dict]:
    rows = []
    for names, unit, scope in ((DAYS, "day", "week, starting from Monday"), (MONTHS, "month", "year")):
        for index, name in enumerate(names):
            after, before, group = names[(index + 1) % len(names)], names[index - 1], f"{unit}:{name}"
            rows += [
                _row(f"What {unit} comes after {name}?", after, "fact", group),
                _row(f"Which {unit} follows {name}? Reply with one word.", after, "fact", group),
                _row(f"What {unit} comes before {name}?", before, "fact", group),
                _row(f"Which {unit} is right before {name}? Reply with one word.", before, "fact", group),
                _row(f"What is {unit} number {index + 1} of the {scope}?", name, "fact", group),
            ]
    rows += [
        _row("Name the days of the week in order, starting with Monday.", ", ".join(DAYS),
             "fact", "day:list"),
        _row("List the months of the year in order.", ", ".join(MONTHS), "fact", "month:list"),
        _row("Which days make up the weekend?", "Saturday and Sunday", "fact", "day:weekend"),
        _row("What is the first month of the year?", "January", "fact", "month:first"),
        _row("What is the last month of the year?", "December", "fact", "month:last"),
    ]
    return rows


def opposite_rows() -> list[dict]:
    rows = []
    for word, opposite in OPPOSITES:
        if {word, opposite} & HOLDOUT_OPPOSITES:
            continue
        for first, second in ((word, opposite), (opposite, word)):
            group = f"opposite:{first}"
            rows += [
                _row(f"What is the opposite of {first}?", second, "english", group),
                _row(f"Give the opposite of {first}. Reply with one word.", second, "english", group),
                _row(f"Name a word that means the opposite of {first}.", second, "english", group),
            ]
    return rows


def plural_rows() -> list[dict]:
    rows = []
    for singular, plural in PLURALS:
        if singular in HOLDOUT_PLURALS:
            continue
        group = f"plural:{singular}"
        rows += [
            _row(f"What is the plural of {singular}?", plural, "english", group),
            _row(f"Give the plural of {singular}. Reply with one word.", plural, "english", group),
            _row(f"One {singular}, two what?", plural, "english", group),
            _row(f"What is the singular of {plural}?", singular, "english", group),
        ]
    return rows


def fact_rows() -> list[dict]:
    rows = []
    for index, (word, sentence, phrasings) in enumerate(FACTS):
        for phrasing in phrasings:
            rows.append(_row(phrasing, sentence, "fact", f"fact:{index}"))
            suffix = " Reply with the number." if word.isdigit() else " Reply with one word."
            rows.append(_row(phrasing + suffix, word, "fact", f"fact:{index}"))
    return rows


def copy_rows() -> list[dict]:
    rows = []
    for word in COPY_WORDS:
        if word in HOLDOUT_COPY:
            continue
        rows += [
            _row(f"Write only the word {word}.", word, "instruction", f"copy:{word}"),
            _row(f"Repeat this word exactly: {word}", word, "instruction", f"copy:{word}"),
            _row(f"Say {word} and nothing else.", word, "instruction", f"copy:{word}"),
        ]
    return rows


def refusal_rows() -> list[dict]:
    rows = []
    for index, (questions, answer) in enumerate(REFUSALS):
        for question in questions:
            for wrapper in REFUSAL_WRAPPERS:
                rows.append(_row(wrapper.replace("{q}", question), answer, "unknown",
                                 f"refusal:{index}"))
    return rows


def short_fact_rows() -> list[dict]:
    """All rows in a fixed order, each {prompt, answer, category, group}, with unique prompts."""
    worded = calendar_rows() + opposite_rows() + plural_rows() + fact_rows() + copy_rows()
    # Arithmetic already has four templates and refusals three wrappers; the word questions
    # get extra framings so they are not outnumbered by arithmetic.
    worded += [dict(row, prompt=wrapper + row["prompt"])
               for wrapper in WORD_WRAPPERS for row in list(worded)]
    rows = arithmetic_rows() + comparison_rows() + worded + refusal_rows()
    unique, seen = [], set()
    for row in rows:
        key = " ".join(row["prompt"].casefold().split())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique
