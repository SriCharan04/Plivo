import json
import random
import re
from typing import Dict, Any, List, Callable

# --- 1. Revised PII Data Pools and Generators ---

FIRST_NAMES = ["Ramesh", "Priya", "Rohit", "Asha", "Sanjay", "Kiran", "Vijay", "Meena", "Alok", "Nisha", "Gaurav"]
SURNAMES = ["Sharma", "Verma", "Singh", "Kumar", "Mehta", "Patel", "Reddy", "Dutt", "Iyer", "Khan", "Choudhary"]
CITIES = ["Mumbai", "Chennai", "Delhi", "Bangalore", "Kolkata", "Pune", "Hyderabad", "Jaipur", "Lucknow", "Ahmedabad"]
LOCATIONS = ["Cyber City Tower 5", "Indira Nagar", "Vasant Kunj", "Hauz Khas", "Connaught Place", "Whitefield", "Jubilee Hills", "DLF Phase 3", "Electronic City", "Koregaon Park"]
DOMAINS = ["gmail", "outlook", "yahoo", "rediffmail", "hotmail", "protonmail"]

# Digit to word map
DIGIT_MAP = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
}

# Expanded Connectors
CONNECTORS = [
    "and also", "and the next piece of info is", "the other detail is",
    "next up is", "in addition to that", "you should also note",
    "and furthermore", "coupled with that", "please note down",
    "as well as"
]

# Expanded Introductory phrases for each PII type
INTRO_PHRASES = {
    "PERSON_NAME": ["my name is", "this is", "I am called", "the customer name is"],
    "CREDIT_CARD": ["my card number is", "the card is", "the credit card details are"],
    "PHONE": ["call me on", "my number is", "the contact phone is", "you can reach me at"],
    "EMAIL": ["my email is", "reach me at", "the email address is"],
    "DATE": ["the date is", "travel on", "the required date of travel is", "the expiry is"],
    "CITY": ["I live in", "I am from", "currently based in", "my city is"],
    "LOCATION": ["the address is", "the delivery point is", "my location is", "the physical address is"]
}

# --- 2. Dynamic PII Generation Functions with Format Randomization ---

def generate_name() -> str:
    """Generates a first name, and randomly includes a surname (50% chance)."""
    first_name = random.choice(FIRST_NAMES)
    
    if random.choice([True, False]):
        surname = random.choice(SURNAMES)
        return f"{first_name} {surname}"
    else:
        return first_name

def generate_credit_card() -> str:
    """Generates a 16-digit card number in either spoken or numeric form."""
    digits = "".join([f"{random.randint(1000, 9999)}" for _ in range(4)])
    
    if random.choice(["spoken", "numeric"]) == "spoken":
        # Spoken form (e.g., 'five five five five three two one zero')
        parts = [DIGIT_MAP[char] for char in digits if char.isdigit()]
        return " ".join(parts)
    else:
        # Numeric form (e.g., '5555 5555 5555 4444')
        return " ".join([digits[i:i+4] for i in range(0, 16, 4)])

def generate_phone() -> str:
    """Generates a 10-digit phone number in either spoken or numeric form."""
    pii_value = f"{random.randint(1000000000, 9999999999)}"
    
    if random.choice(["spoken", "numeric"]) == "spoken":
        # Spoken form (e.g., 'nine one two three four...')
        parts = [DIGIT_MAP[char] for char in pii_value if char.isdigit()]
        return " ".join(parts)
    else:
        # Numeric form (e.g., '9876543210')
        # Introduce random pauses/separators for more noise
        separators = random.choice(["", " ", "-", " "])
        if separators == "": # 50% chance of being continuous or slightly separated
             return pii_value
        else:
             return f"{pii_value[:5]}{separators}{pii_value[5:]}"


def generate_email() -> str:
    """Generates a spoken email address (e.g., first dot last at domain dot com)."""
    name = generate_name().replace(" ", "")
    domain = random.choice(DOMAINS)
    
    # Email always uses the noisy, spoken format to simulate STT issues
    return f"{name} at {domain} dot com"

def generate_date() -> str:
    """Generates a date in either spoken or numeric form."""
    day = f"{random.randint(1, 28):02d}"
    month = f"{random.randint(1, 12):02d}"
    year = random.choice(["2024", "2025", "2026"])
    
    if random.choice(["spoken", "numeric"]) == "spoken":
        # Spoken form (e.g., 'zero two eleven twenty twenty five')
        pii_value = f"{day} {month} {year}"
        return " ".join([DIGIT_MAP.get(char, char) for char in pii_value if char.isalnum()])
    else:
        # Numeric form (e.g., '02/11/2025' or '02 11 2025')
        separator = random.choice([" ", "/", ""])
        if separator == "":
             return f"{day}{month}{year}"
        else:
             return f"{day}{separator}{month}{separator}{year}"


# Mapping of PII types to their generation function
PII_GENERATORS: Dict[str, Callable[[], str]] = {
    "PERSON_NAME": generate_name,
    "EMAIL": generate_email,
    "CREDIT_CARD": generate_credit_card,
    "PHONE": generate_phone,
    "CITY": lambda: random.choice(CITIES),
    "LOCATION": lambda: random.choice(LOCATIONS),
    "DATE": generate_date
}

# --- 3. Dynamic Sentence Construction (Re-used Logic) ---

def generate_dynamic_example(id_num: int) -> Dict[str, Any]:
    """Generates a single dynamic JSONL example with correct character offsets."""
    
    # Randomly select the number of entities (1 to 5)
    num_entities = random.randint(1, 5)
    
    # Randomly choose PII types to include
    all_pii_types = list(PII_GENERATORS.keys())
    chosen_types = random.sample(all_pii_types, min(num_entities, len(all_pii_types)))
    
    final_text = ""
    current_char_offset = 0
    entities = []
    
    for i, label in enumerate(chosen_types):
        pii_value = PII_GENERATORS[label]()
        
        if i == 0:
            # Start of sentence
            phrase = random.choice(INTRO_PHRASES[label]) + " "
            final_text += phrase
            current_char_offset += len(phrase)
        else:
            # Connector phrase to link to the next PII
            connector_phrase = random.choice(CONNECTORS)
            intro_phrase = random.choice(INTRO_PHRASES[label])
            # Structure: ", and also the contact phone is [PHONE]"
            phrase = f", {connector_phrase} {intro_phrase} "
            final_text += phrase
            current_char_offset += len(phrase)
            
        # PII value insertion
        start = current_char_offset
        final_text += pii_value
        end = current_char_offset + len(pii_value)
        
        entities.append({
            "start": start,
            "end": end,
            "label": label
        })
        current_char_offset = end
        
    return {
        "id": f"utt_{id_num:04d}",
        "text": final_text.capitalize() + ".",
        "entities": entities
    }

# --- 4. Main Generation Loop ---

def generate_dataset(num_train: int, num_dev: int):
    """Generates and writes the train and dev datasets to JSONL files."""
    
    train_data = []
    dev_data = []
    
    # Generate Train data (starting ID 1)
    for i in range(1, num_train + 1):
        train_data.append(generate_dynamic_example(i))

    # Generate Dev data (starting ID 1001 to avoid ID overlap)
    for i in range(1001, 1001 + num_dev):
        dev_data.append(generate_dynamic_example(i))

    # Write to files
    with open("train_unbiased.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open("dev_unbiased.jsonl", "w") as f:
        for item in dev_data:
            f.write(json.dumps(item) + "\n")

    print(f"Generated {len(train_data)} training examples into 'train_unbiased.jsonl'")
    print(f"Generated {len(dev_data)} development examples into 'dev_unbiased.jsonl'")

# Execute generation
generate_dataset(num_train=5000, num_dev=1500)