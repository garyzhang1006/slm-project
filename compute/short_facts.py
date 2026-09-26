"""Project-authored short questions with short answers, released as CC0-1.0.

Each fact appears in several phrasings so the model learns the fact rather than one string.
The items tested by data/simple_questions_holdout.json (4 plus 9, days in a week, the
opposite of tall, a home address, and so on) are left out on purpose, so a holdout gain
measures transfer to unseen questions instead of recall of trained ones. The same holds for
data/everyday_eval.json: its entities (capitals, opposites, plurals, names in its stories) and its
question wordings are kept out of these rows, and tests/test_everyday_eval.py checks both.
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

# Countries, languages, nouns, verbs and names below avoid the entities in data/everyday_eval.json.
CAPITALS = (
    ("Poland", "Warsaw"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"),
    ("Austria", "Vienna"), ("Hungary", "Budapest"), ("the Netherlands", "Amsterdam"),
    ("Belgium", "Brussels"), ("the Czech Republic", "Prague"), ("Turkey", "Ankara"),
    ("India", "New Delhi"), ("Thailand", "Bangkok"), ("South Korea", "Seoul"), ("Chile", "Santiago"),
    ("Colombia", "Bogota"), ("Cuba", "Havana"), ("the Philippines", "Manila"), ("Vietnam", "Hanoi"),
    ("Ghana", "Accra"), ("Nigeria", "Abuja"), ("Ethiopia", "Addis Ababa"), ("Morocco", "Rabat"),
    ("New Zealand", "Wellington"), ("Argentina", "Buenos Aires"), ("Iceland", "Reykjavik"),
    ("Scotland", "Edinburgh"), ("Wales", "Cardiff"), ("Jamaica", "Kingston"), ("Ukraine", "Kyiv"),
)
LANGUAGES = (
    ("France", "French"), ("Italy", "Italian"), ("Japan", "Japanese"), ("Spain", "Spanish"),
    ("Poland", "Polish"), ("Sweden", "Swedish"), ("the Netherlands", "Dutch"), ("Turkey", "Turkish"),
)
KINDS = {
    "fruit": ("grape", "pear", "plum", "melon", "kiwi", "apricot", "pineapple", "mango"),
    "vegetable": ("onion", "cabbage", "lettuce", "spinach", "broccoli", "celery", "pea", "potato"),
    "animal": ("frog", "camel", "squirrel", "fox", "tiger", "wolf", "rabbit", "owl"),
    "tool": ("hammer", "saw", "screwdriver", "wrench", "drill", "shovel", "rake"),
    "clothing": ("scarf", "sock", "jacket", "sweater", "skirt", "hat", "belt"),
    "vehicle": ("truck", "boat", "airplane", "tractor", "taxi", "van", "helicopter"),
}
# (base, past tense). Regular and irregular verbs, each past form standard in all English varieties.
PAST_TENSES = (
    ("go", "went"), ("eat", "ate"), ("run", "ran"), ("see", "saw"), ("swim", "swam"),
    ("sing", "sang"), ("write", "wrote"), ("drink", "drank"), ("take", "took"), ("give", "gave"),
    ("come", "came"), ("sit", "sat"), ("stand", "stood"), ("sleep", "slept"), ("find", "found"),
    ("buy", "bought"), ("bring", "brought"), ("think", "thought"), ("teach", "taught"),
    ("fly", "flew"), ("grow", "grew"), ("know", "knew"), ("draw", "drew"), ("speak", "spoke"),
    ("break", "broke"), ("choose", "chose"), ("wear", "wore"), ("begin", "began"),
    ("forget", "forgot"), ("make", "made"), ("tell", "told"), ("hold", "held"), ("keep", "kept"),
    ("feel", "felt"), ("meet", "met"), ("win", "won"), ("ride", "rode"), ("cut", "cut"),
    ("put", "put"), ("walk", "walked"), ("jump", "jumped"), ("play", "played"), ("cook", "cooked"),
    ("open", "opened"), ("clean", "cleaned"), ("help", "helped"), ("dance", "danced"),
    ("smile", "smiled"), ("stop", "stopped"), ("carry", "carried"), ("cry", "cried"),
    ("try", "tried"), ("plan", "planned"), ("hop", "hopped"), ("laugh", "laughed"), ("wash", "washed"),
)
# (adjective, comparative, superlative).
DEGREES = (
    ("big", "bigger", "biggest"), ("small", "smaller", "smallest"), ("fast", "faster", "fastest"),
    ("slow", "slower", "slowest"), ("old", "older", "oldest"), ("young", "younger", "youngest"),
    ("happy", "happier", "happiest"), ("sad", "sadder", "saddest"), ("warm", "warmer", "warmest"),
    ("cold", "colder", "coldest"), ("hot", "hotter", "hottest"), ("good", "better", "best"),
    ("bad", "worse", "worst"), ("easy", "easier", "easiest"), ("funny", "funnier", "funniest"),
    ("busy", "busier", "busiest"), ("early", "earlier", "earliest"), ("strong", "stronger", "strongest"),
    ("weak", "weaker", "weakest"), ("loud", "louder", "loudest"), ("clean", "cleaner", "cleanest"),
    ("dark", "darker", "darkest"), ("soft", "softer", "softest"), ("high", "higher", "highest"),
    ("low", "lower", "lowest"), ("deep", "deeper", "deepest"), ("wide", "wider", "widest"),
    ("thin", "thinner", "thinnest"), ("rich", "richer", "richest"), ("sweet", "sweeter", "sweetest"),
    ("kind", "kinder", "kindest"), ("brave", "braver", "bravest"), ("nice", "nicer", "nicest"),
)
# The article follows the first sound, so "hour" takes "an" and "uniform" takes "a".
ARTICLES = (
    ("an", "ant"), ("an", "igloo"), ("an", "elbow"), ("an", "island"), ("an", "umbrella"),
    ("an", "orange"), ("an", "hour"), ("an", "engine"), ("an", "arm"), ("an", "idea"),
    ("an", "oven"), ("an", "uncle"), ("an", "actor"), ("an", "insect"), ("an", "eagle"),
    ("an", "iceberg"), ("a", "cat"), ("a", "ball"), ("a", "house"), ("a", "lamp"), ("a", "river"),
    ("a", "pencil"), ("a", "unicorn"), ("a", "uniform"), ("a", "university"), ("a", "garden"),
    ("a", "door"), ("a", "cloud"), ("a", "table"), ("a", "drum"), ("a", "ship"), ("a", "flag"),
    ("a", "nest"), ("a", "kettle"),
)
# Holdout rows 19 and 24 sort pear, apple, banana and ask len("sun"), so those words stay out, and the
# phrasings differ from the holdout templates so the rows teach the skill without copying the questions.
HOLDOUT_SORT_WORDS = {"pear", "apple", "banana", "sun"}
SORT_WORDS = ("cherry", "grape", "lemon", "mango", "olive", "peach", "plum", "kiwi", "melon", "tiger",
              "zebra", "horse", "rabbit", "eagle", "otter", "bread", "cheese", "butter", "honey", "rice",
              "river", "cloud", "forest", "desert", "island", "violin", "drum", "flute", "piano", "guitar")
STORY_NAMES = ("Alice", "Ivan", "Chen", "Sofia", "Amir", "Beth", "Diego", "Kira")
STORY_ITEMS = ("stickers", "crayons", "shells", "buttons", "cookies", "balloons")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TEENS = ("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS_WORDS = ("twenty thirty forty fifty sixty seventy eighty ninety").split()

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


def number_in_words(number: int) -> str:
    """English words for 0 to 100, hyphenating compound tens ("forty-two")."""
    if number == 100:
        return "one hundred"
    if number < 20:
        return (NUMBER_WORDS[:10] + _TEENS)[number]
    tens, units = divmod(number, 10)
    word = _TENS_WORDS[tens - 2]
    return f"{word}-{NUMBER_WORDS[units]}" if units else word


def number_rows() -> list[dict]:
    rows = []
    for number in range(101):
        group, word = f"number:{number}", number_in_words(number)
        rows += [
            _row(f"Write the number {number} in words.", word, "math", group),
            _row(f"Which number is written as {word}? Reply with digits.", str(number), "math", group),
            _row(f"Is {number} odd or even?", "even" if number % 2 == 0 else "odd", "math", group),
        ]
        if number < 100:
            rows.append(_row(f"What number comes right after {number}?", str(number + 1), "math", group))
            rows.append(_row(f"Count up by one from {number}. What comes next?", str(number + 1), "math",
                             group))
        if number > 0:
            rows.append(_row(f"What number comes just before {number}?", str(number - 1), "math", group))
    # 16 is skipped because data/everyday_eval.json asks 16 + 16.
    for number in (value for value in range(1, 51) if value != 16):
        rows += [
            _row(f"What is double {number}?", str(2 * number), "math", f"double:{number}"),
            _row(f"What is half of {2 * number}?", str(number), "math", f"double:{number}"),
        ]
    return rows


def clock_rows() -> list[dict]:
    rows = []
    for hour in range(1, 13):
        for delta in range(1, 7):
            # data/everyday_eval.json asks noon plus 2 hours and 9 until noon.
            if (hour, delta) in ((12, 2), (9, 3)):
                continue
            later, earlier = (hour + delta - 1) % 12 + 1, (hour - delta - 1) % 12 + 1
            group = f"clock:{hour}"
            rows += [
                _row(f"It is {hour} o'clock now. What time will it be in {delta} hours?",
                     f"{later} o'clock", "math", group),
                _row(f"It is {hour} o'clock now. What time was it {delta} hours ago?",
                     f"{earlier} o'clock", "math", group),
            ]
    return rows


def story_rows() -> list[dict]:
    rows = []
    for a in range(2, 13):
        for b in range(1, 10):
            name = STORY_NAMES[(a + b) % len(STORY_NAMES)]
            items = STORY_ITEMS[(a * b) % len(STORY_ITEMS)]
            rows.append(_row(f"{name} has {a} {items} and finds {b} more. "
                             f"How many {items} does {name} have now?", str(a + b), "math",
                             f"story:add:{a}:{b}"))
            if b < a:
                rows.append(_row(f"{name} had {a} {items} and lost {b} of them. How many {items} are left?",
                                 str(a - b), "math", f"story:sub:{a}:{b}"))
    return rows


def letter_rows() -> list[dict]:
    rows = []
    for index, letter in enumerate(LETTERS):
        group = f"letter:{letter}"
        if index + 1 < len(LETTERS):
            rows.append(_row(f"Which letter comes after {letter} in the alphabet?", LETTERS[index + 1],
                             "english", group))
        if index > 0:
            rows.append(_row(f"Which letter comes before {letter} in the alphabet?", LETTERS[index - 1],
                             "english", group))
        if letter != "Y":  # Y can be either, so it gets no vowel question.
            rows.append(_row(f"Is the letter {letter} a vowel or a consonant?",
                             "vowel" if letter in "AEIOU" else "consonant", "english", group))
    return rows


def world_rows() -> list[dict]:
    rows = []
    for country, city in CAPITALS:
        group, title = f"capital:{country}", country[0].upper() + country[1:]
        rows += [
            _row(f"What is the capital of {country}?", city, "fact", group),
            _row(f"Which city is the capital of {country}?", f"{city} is the capital of {country}.", "fact",
                 group),
            _row(f"{city} is the capital of which country?", title, "fact", group),
        ]
    for country, language in LANGUAGES:
        group = f"language:{country}"
        rows += [
            _row(f"What language do most people in {country} speak?", language, "fact", group),
            _row(f"Which language is mainly spoken in {country}?", language, "fact", group),
        ]
    kinds = list(KINDS)
    for position, (kind, items) in enumerate(KINDS.items()):
        for item in items:
            article = "an" if item[0] in "aeiou" else "a"
            group = f"kind:{item}"
            rows.append(_row(f"Which group does {article} {item} belong to: fruit, vegetable, animal, tool, "
                             "clothing or vehicle?", kind, "fact", group))
            for step in (1, 3):
                other = kinds[(position + step) % len(kinds)]
                rows.append(_row(f"Is {article} {item} a kind of {kind} or a kind of {other}?", kind, "fact",
                                 group))
                rows.append(_row(f"Is {article} {item} a kind of {other} or a kind of {kind}?", kind, "fact",
                                 group))
    return rows


def grammar_rows() -> list[dict]:
    rows = []
    for base, past in PAST_TENSES:
        group = f"past:{base}"
        rows += [
            _row(f"What is the past tense of {base}?", past, "english", group),
            _row(f"Change {base} to the past tense.", past, "english", group),
            _row(f"Today I {base}. Yesterday I ___. Fill in the blank.", past, "english", group),
        ]
    for adjective, comparative, superlative in DEGREES:
        group = f"degree:{adjective}"
        rows += [
            _row(f"What is the comparative form of {adjective}?", comparative, "english", group),
            _row(f"What is the superlative form of {adjective}?", superlative, "english", group),
            _row(f"Complete the pattern: {adjective}, {comparative}, ___", superlative, "english", group),
        ]
    for article, noun in ARTICLES:
        group = f"article:{noun}"
        rows += [
            _row(f"Fill in a or an: ___ {noun}", article, "english", group),
            _row(f"Should you say a {noun} or an {noun}?", f"{article} {noun}", "english", group),
        ]
    return rows


def sort_rows() -> list[dict]:
    words = [word for word in SORT_WORDS if word not in HOLDOUT_SORT_WORDS]
    rows = []
    # Consecutive triples, each listed out of order by a fixed rotation, so the answer is never the prompt.
    for index in range(len(words)):
        triple = [words[index], words[(index + 7) % len(words)], words[(index + 13) % len(words)]]
        shown = triple[1:] + triple[:1] if sorted(triple) == triple else triple
        listed, answer = ", ".join(shown), ", ".join(sorted(triple))
        group = f"sort:{answer}"
        rows += [
            _row(f"Sort these words from A to Z: {listed}.", answer, "english", group),
            _row(f"Arrange alphabetically: {listed}", answer, "english", group),
        ]
    for index in range(len(words)):
        left, right = words[index], words[(index + 11) % len(words)]
        if left[0] == right[0]:
            continue
        rows.append(_row(f"Which word comes first in the alphabet, {left} or {right}?", min(left, right),
                         "english", f"first:{min(left, right)}:{max(left, right)}"))
    return rows


def code_rows() -> list[dict]:
    rows = []
    for word in (word for word in SORT_WORDS if word not in HOLDOUT_SORT_WORDS):
        group = f"code:str:{word}"
        rows += [
            _row(f'How many characters are in the Python string "{word}"?', str(len(word)), "code", group),
            _row(f'What is len("{word}") in Python?', str(len(word)), "code", group),
            _row(f'In Python, what does "{word}".upper() give?', f'"{word.upper()}"', "code", group),
        ]
    for size in range(1, 7):
        items = ", ".join(str(number) for number in range(2, 2 + size))
        rows.append(_row(f"What is len([{items}]) in Python?", str(size), "code", f"code:list:{size}"))
    return rows

def short_fact_rows() -> list[dict]:
    """All rows in a fixed order, each {prompt, answer, category, group}, with unique prompts."""
    worded = (calendar_rows() + opposite_rows() + plural_rows() + fact_rows() + copy_rows()
              + world_rows() + grammar_rows())
    # Arithmetic already has four templates and refusals three wrappers; the word questions
    # get extra framings so they are not outnumbered by arithmetic.
    worded += [dict(row, prompt=wrapper + row["prompt"])
               for wrapper in WORD_WRAPPERS for row in list(worded)]
    rows = (arithmetic_rows() + comparison_rows() + worded + refusal_rows() + number_rows()
            + clock_rows() + story_rows() + letter_rows() + sort_rows() + code_rows())
    unique, seen = [], set()
    for row in rows:
        key = " ".join(row["prompt"].casefold().split())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique
