import json
import openai
import random
import os

# Determine the base directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the credentials
secrets_path = os.path.join(base_dir, 'LERACS', 'LERACS_CONTROL', 'secrets.json')
with open(secrets_path) as f:
    credentials = json.load(f)

# Load the personas from the JSON file
personas_path = os.path.join(base_dir, 'personas.json')
with open(personas_path, 'r') as file:
    personas = json.load(file)

# BFI-10 questions
bfi_10_questions = [
    "gets nervous easily.",
    "tends to find fault with others.",
    "is outgoing, sociable.",
    "is generally trusting.",
    "tends to be lazy.",
    "is relaxed, handles stress well.",
    "has few artistic interests.",
    "does a thorough job.",
    "is reserved.",
    "has an active imagination."
]

# OpenAI API key
openai.api_key = credentials["openai"]["OPENAI_API_KEY"]

# Function to create the prompt for a persona
def create_prompt(persona):
    description = f"{persona['age']}, {persona['gender']}, {persona['occupation']} - {persona['description']}"
    questions_order = random.sample(bfi_10_questions, len(bfi_10_questions))
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions_order)])
    prompt = (
        f"Imagine the following person (age, gender, description): {description}\n"
        "Rate this person / complete the questionnaire for this person, on a scale of 1 (disagree strongly) to 5 (agree strongly).\n"
        "Report in the following format on a single line, e.g., 5 1 4 ...\n"
        "Report only digits, nothing else.\n"
        f"{questions_text}"
    )
    return prompt, questions_order

# Function to submit the prompt to OpenAI API and get the response
def get_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Using the "gpt-4" model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message['content'].strip()

# Process each persona
results = {}
for persona_id, persona in personas.items():
    persona_results = []
    for _ in range(1):  # Each persona is prompted once
        prompt, questions_order = create_prompt(persona)
        response = get_response(prompt)
        persona_results.append({
            "response": response,
            "questions_order": questions_order
        })
    results[persona_id] = persona_results

# Save the results to a JSON file
results_path = os.path.join(base_dir, 'persona_bfi_results.json')
with open(results_path, 'w') as outfile:
    json.dump(results, outfile, indent=4)

print("BFI-10 questionnaire completed for all personas.")

