import openai
import json
import re

# Determine the base directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the credentials
secrets_path = os.path.join(base_dir, 'LERACS', 'LERACS_CONTROL', 'secrets.json')
with open(secrets_path) as f:
    credentials = json.load(f)

# Generate personas using the OpenAI API
def generate_personas(api_key, num_personas=25, output_file='personas.json'):
    openai.api_key = api_key
    prompt = (
        f"Please give me {num_personas} realistic and diverse personas with realistic names and realistic personality descriptions.\n"
        "Also add their age and gender. The persona should be described by means of three brief sentences separated by semicolons.\n"
        "Report each persona on a single line, numbered 0001 to 0025. Separate age, gender, profession/activity/job versus description of who they are by means of a dash.\n"
        "These personas are going to test a robotic system using AI, where the persona has to chat with the robot."
        f"Make sure that from the {num_personas} personas that there are 5 saboteurs, 5 cognitively limited persons (for example a child), 5 technician experts, 5 technician newcomers and 5 normally generated personas. \n"
        f"Say behind every name what they are from these 5 categories\n"
        "Only personas; nothing else."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a persona generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=1
    )

    personas_text = response.choices[0]['message']['content']

    try:
        # Extract persona lines from the response text
        persona_lines = personas_text.strip().split('\n')

        # Convert persona lines to JSON
        personas_dict = {}
        for line in persona_lines:
            match = re.match(r'^(\d{4}) - (.+)$', line.strip())
            if match:
                persona_id = match.group(1)
                details = match.group(2).split(";", 2)
                age_gender_profession = details[0].split(", ")
                name = age_gender_profession[0]
                age = int(age_gender_profession[1])
                gender = age_gender_profession[2]
                profession = details[1].strip()
                description = details[2].strip()

                personas_dict[persona_id] = {
                    "Name": name,
                    "Age": age,
                    "Gender": gender,
                    "Profession": profession,
                    "Description": description
                }

        if not personas_dict:
            print("Failed to generate personas. Response content might not be in the expected format.")
            print("Response content:", personas_text)
            return []

        # Save personas to a file
        with open(output_file, 'w') as f:
            json.dump(personas_dict, f, indent=4)

        return personas_dict

    except json.JSONDecodeError as e:
        print("Failed to decode JSON. Response content might not be in JSON format.")
        print("Response content:", personas_text)
        print("Error message:", str(e))
        return []

def main():
    openai.api_key = credentials["openai"]["OPENAI_API_KEY"]

    # Generate personas
    personas = generate_personas(openai.api_key)

if __name__ == "__main__":
    main()
