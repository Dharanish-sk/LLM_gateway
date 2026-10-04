"""Versioned evaluation seed data. Export with `gateway-eval datasets`.

These modest, mostly templated suites are regression/profiling datasets, not a claim
of broad domain coverage. Add independently authored families for deployment.
"""

DATA_VERSION = "2026-10-v1"


def prompts() -> list[dict]:
    rows = []
    factual = [
        ("What is the capital of Australia?", "Canberra"),
        ("What is the chemical symbol for gold?", "Au"),
        ("How many sides does a hexagon have?", "Six"),
        ("Which planet is closest to the Sun?", "Mercury"),
        ("At standard atmospheric pressure, at what Celsius temperature does pure water freeze?", "0 degrees Celsius"),
        ("Who wrote Pride and Prejudice?", "Jane Austen"),
        ("What is the largest ocean on Earth by area?", "Pacific Ocean"),
        ("What organ pumps blood around the human body?", "The heart"),
        ("How many bytes are in a kibibyte (KiB)?", "1024"),
        ("Which gas do plants take from the atmosphere during photosynthesis?", "Carbon dioxide"),
    ]
    summaries = [
        ("The library will close on Monday for repairs. It reopens Tuesday at 9 am. Online loans remain available.", "The library closes Monday for repairs, reopens Tuesday at 9 am, and keeps online loans available."),
        ("A team tested 100 batteries. Eight failed inspection. The remaining 92 were shipped on Friday.", "Eight of 100 batteries failed; 92 passed and shipped Friday."),
        ("The council approved a park renovation. Work begins in June and ends in August. The playground remains open.", "The park renovation runs June through August while the playground stays open."),
        ("Sales rose from 200 to 250 units. Revenue fell because the company reduced prices. Management plans to review discounts.", "Unit sales rose but discounts reduced revenue, prompting a review."),
        ("The train is delayed by 20 minutes due to a signal fault. Passengers may use their tickets on the next bus.", "A signal fault delays the train 20 minutes; train tickets are valid on the next bus."),
        ("Researchers surveyed 60 volunteers. Most preferred morning exercise. The sample was not representative of the wider population.", "Most of 60 surveyed volunteers preferred morning exercise, but the sample was not representative."),
        ("Version 2 fixes a login bug and adds export support. Older plugins may break. Users should back up settings before updating.", "Version 2 fixes login and adds export, but may break older plugins; back up settings first."),
        ("The clinic offers free vaccinations on Saturday from 10 to 2. Booking is required. Children must attend with a guardian.", "Book for free Saturday vaccinations from 10 to 2; children need a guardian."),
        ("A storm damaged two bridges. The eastern route is closed. The western route has reopened to light vehicles only.", "Storm damage closes the eastern bridge route; the western route allows only light vehicles."),
        ("The museum acquired 12 paintings. Nine go on display next month. Three need restoration before they can be shown.", "Nine of 12 acquired paintings will be displayed next month; three need restoration."),
    ]
    reasoning = [
        ("Reason briefly: A train travels 120 km in 2 hours, then 90 km in 1 hour. What is its average speed?", "210 km / 3 hours = 70 km/h, not the unweighted mean of speeds."),
        ("Reason briefly: All tulips are flowers. Some flowers are red. Must some tulips be red?", "No. The red flowers need not include any tulips."),
        ("Reason briefly: A $80 item is discounted 25%, then taxed 10%. What is the final price?", "$80 * .75 * 1.10 = $66."),
        ("Reason briefly: A box has 3 red and 2 blue balls. Two are drawn without replacement. Probability both are red?", "(3/5)*(2/4) = 3/10."),
        ("Reason briefly: Alice is older than Bob. Bob is older than Chen. Who is youngest?", "Chen, by transitivity."),
        ("Reason briefly: Find the smallest integer x such that 3x + 2 > 14.", "x > 4, so the smallest integer is 5."),
        ("Reason briefly: Two workers each make 4 parts per hour. How many parts do they make together in 3 hours?", "2 * 4 * 3 = 24 parts."),
        ("Reason briefly: A rectangle has perimeter 30 cm and width 5 cm. Find its area.", "Length is 15 - 5 = 10 cm; area is 50 square cm."),
        ("Reason briefly: A statement says 'If it rains, the ground gets wet.' The ground is wet. Does that prove rain?", "No, other causes can wet the ground; affirming the consequent is invalid."),
        ("Reason briefly: A sequence starts 2, 6, 12, 20, 30. Using the pattern n(n+1), what is next?", "6 * 7 = 42."),
    ]
    for category, pairs in (("factual", factual), ("summarization", summaries), ("reasoning", reasoning)):
        for index, (text, reference) in enumerate(pairs):
            prompt = "Summarize this in one sentence, preserving important constraints: " + text if category == "summarization" else text
            rows.append({"id": f"{category}-{index:02}", "family": f"{category}-{index:02}", "type": category, "prompt": prompt, "reference": reference})
    people = [("Asha", 24, "Pune"), ("Ben", 31, "Leeds"), ("Cara", 19, "Lima"), ("Diego", 42, "Quito"), ("Eve", 28, "Oslo"), ("Femi", 35, "Lagos"), ("Gita", 22, "Delhi"), ("Hugo", 50, "Porto"), ("Iris", 27, "Rome"), ("Jin", 38, "Seoul")]
    for index, (name, age, city) in enumerate(people):
        expected = {"name": name, "age": age, "city": city}
        rows.append({
            "id": f"json-{index:02}", "family": "json-person-extraction", "type": "json",
            "prompt": f"Extract name, age and city as a JSON object, with no markdown or extra keys. Text: {name} is {age} years old and lives in {city}.",
            "reference": expected, "schema": {
                "type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}},
                "required": ["name", "age", "city"], "additionalProperties": False,
            },
        })
    return rows


def cache_pairs() -> list[dict]:
    triples = []
    countries = [("Australia", "Austria"), ("Slovakia", "Slovenia"), ("Niger", "Nigeria"), ("Iran", "Iraq"), ("Latvia", "Lithuania"), ("Sweden", "Switzerland"), ("Georgia", "Germany"), ("India", "Indonesia"), ("Mali", "Malta"), ("Guinea", "Guinea-Bissau")]
    for country, other in countries:
        triples.append(("capital", f"What is the capital of {country}?", f"Name {country}'s capital city.", f"What is the capital of {other}?"))
    for number in range(1, 11):
        triples.extend([
            ("units", f"Convert {number} kilometers to meters.", f"How many meters equal {number} km?", f"Convert {number} miles to meters."),
            ("arithmetic", f"What is {number} plus 12?", f"Calculate the sum of {number} and 12.", f"What is {number} minus 12?"),
            ("percentage", f"Find 15 percent of {number * 100}.", f"Calculate 15% of {number * 100}.", f"Find 50 percent of {number * 100}."),
            ("negation", f"List integers from 1 to {number + 10} that are even.", f"Which even integers are between 1 and {number + 10}, inclusive?", f"List integers from 1 to {number + 10} that are not even."),
            ("sorting", f"Sort {number}, {number + 5}, {number + 2} in ascending order.", f"Order these numbers smallest first: {number}, {number + 5}, {number + 2}.", f"Sort {number}, {number + 5}, {number + 2} in descending order."),
            ("time", f"What is {number}:00 UTC in UTC+2?", f"Convert {number}:00 from UTC to a timezone two hours ahead.", f"What is {number}:00 UTC in UTC-2?"),
            ("schema", f'Extract only name as JSON from "name=Person{number}; age={number + 20}".', f'Return a JSON object with just the name field from "name=Person{number}; age={number + 20}".', f'Extract only age as JSON from "name=Person{number}; age={number + 20}".'),
            ("comparison", f"Is {number} strictly greater than {number}?", f"Does {number} exceed {number}?", f"Is {number} greater than or equal to {number}?"),
            ("temperature", f"Convert {number * 10} degrees Celsius to Fahrenheit.", f"What Fahrenheit temperature equals {number * 10} C?", f"Convert {number * 10} degrees Fahrenheit to Celsius."),
        ])
    rows = []
    for index, (kind, anchor, paraphrase, miss) in enumerate(triples):
        for label, candidate in ((True, paraphrase), (False, miss)):
            rows.append({"id": f"pair-{index:03}-{'positive' if label else 'negative'}", "group": f"anchor-{index:03}", "family": kind, "anchor": anchor, "candidate": candidate, "paraphrase": label})
    return rows
