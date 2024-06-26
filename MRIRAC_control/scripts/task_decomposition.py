import openai
import tiktoken
import json
import os
import re
import argparse
import rospy
import sys
import base64

# Initialize encoding
enc = tiktoken.get_encoding("cl100k_base")

# Load credentials from secrets file
with open('secrets.json') as f:
    credentials = json.load(f)

# Define directory paths
dir_system = './system'
dir_prompt = './prompt'
dir_query = './query'

# Define the order in which prompts will be loaded
prompt_load_order = [
    'prompt_role',
    'input_environment',
    'prompt_function',
    'prompt_environment',
    'prompt_output_format',
    'prompt_example'
]


class ChatGPT:
    """
    The ChatGPT class facilitates interaction between OpenAI's GPT models and robotic systems for environmental interpretation and task execution.
    It processes textual and image data to generate and execute task sequences based on environmental understanding and user instructions.

    Functions:
    - encode_image_to_base64(): Encodes images to base64 format for processing.
    - extract_json_part(): Extracts the JSON formatted string embedded within a larger text block.
    - create_prompt(): Dynamically creates prompts for interaction with GPT models.
    - generate_environment(): Generates environmental descriptions from images using GPT-4 vision capabilities.
    - generate(): Processes instructions and feedback, using GPT-3.5 models to generate textual responses or actions based on the current environment and user input.
    - dump_json(): Serializes and saves the structured data (typically the environment or action outcomes) to a JSON file.
    """

    def __init__(self, credentials, prompt_load_order):
        """
        Initialize the ChatGPT object with necessary credentials and load prompts.

        Args:
            credentials (dict): API credentials for OpenAI.
            prompt_load_order (list): Order in which prompt files will be loaded.
        """
        openai.api_key = credentials["openai"]["OPENAI_API_KEY"]
        self.messages = []
        self.max_token_length = 8000
        self.max_completion_length = 1000
        self.last_response = None
        self.query = ''
        self.instruction = ''

        # Load system message
        fp_system = os.path.join(dir_system, 'system.txt')
        with open(fp_system) as f:
            data = f.read()
        self.system_message = {"role": "system", "content": data}

        # Load ordered prompts
        for prompt_name in prompt_load_order:
            fp_prompt = os.path.join(dir_prompt, prompt_name + '.txt')
            with open(fp_prompt) as f:
                data = f.read()
            data_split = re.split(r'\[user\]\n|\[assistant\]\n', data)
            data_split = [item for item in data_split if len(item) != 0]
            assert len(data_split) % 2 == 0
            for i, item in enumerate(data_split):
                role = "user" if i % 2 == 0 else "assistant"
                self.messages.append({"role": role, "content": item})

        # Load query
        fp_query = os.path.join(dir_query, 'query.txt')
        with open(fp_query) as f:
            self.query = f.read()

    def encode_image_to_base64(self, image_path):
        """
        Encodes an image file to a base64 string.

        Parameters:
            image_path (str): The file path to the image that needs to be encoded.

        Returns:
            str: A base64-encoded string representation of the image.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            print(f"Encoded Image: {encoded_string[:100]}...")
            return encoded_string

    def load_and_format_content(self, file_path, **format_kwargs):
        """
        Loads and formats content from a file with optional keyword formatting.

        Parameters:
            file_path (str): The path to the file to be loaded.
            format_kwargs (dict): Optional keyword arguments for formatting the file content.

        Returns:
            str: The formatted content of the file.
        """
        with open(file_path, 'r') as file:
            content = file.read()
        try:
            formatted_content = content.format(**format_kwargs)
        except KeyError as e:
            # Debug output to help diagnose formatting issues
            print(f"Formatting error: {e}, check for unescaped braces or missing keys")
            print("Problematic content starts with:", content[:500])  # Adjust range as needed
            formatted_content = content  # Optionally use raw content if formatting fails
        return formatted_content

    def create_prompt(self, encoded_image=None):
        """
        Constructs a prompt for the GPT model by aggregating system messages and user or assistant messages stored within the class.
        This method ensures the prompt is within the maximum token length allowed by the model, truncating older messages if necessary to fit the limit.

        Returns:
            list: A list of dictionaries, each representing a message with a 'role' (indicating who the message is from) and 'content' (the message text itself).
        """
        prompt = [self.system_message]
        prompt.extend(self.messages)

        # Add image content to the prompt if provided
        if encoded_image:
            image_prompt = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.instruction
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
            prompt.append(image_prompt)

        # Flatten the prompt contents for length checking
        prompt_content = " ".join(
            [msg["content"] if isinstance(msg["content"], str) else " ".join(
                [c["text"] if isinstance(c, dict) and "text" in c else "" for c in msg["content"]]) for msg in prompt]
        )

        # Truncate the prompt by removing the oldest messages if it exceeds the token limit
        if len(enc.encode(prompt_content)) > self.max_token_length - self.max_completion_length:
            print('Prompt too long, truncating.')
            while len(enc.encode(prompt_content)) > self.max_token_length - self.max_completion_length and len(
                    self.messages) > 2:
                self.messages.pop(0)
                prompt_content = " ".join(
                    [msg["content"] if isinstance(msg["content"], str) else " ".join(
                        [c["text"] if isinstance(c, dict) and "text" in c else "" for c in msg["content"]]) for msg in
                     [self.system_message] + self.messages]
                )

        return prompt

    def extract_json_part(self, text):
        """
        Extracts a JSON-formatted string from a larger text block, ensuring it is clean and parseable.

        Parameters:
            text (str): The text from which to extract the JSON part.

        Returns:
            str: The extracted and cleaned JSON string.
        """
        try:
            start = text.index('{')  # Find the start of JSON object
            end = text.rindex('}') + 1  # Find the end of JSON object, include the closing bracket
            return text[start:end]  # Extract the JSON substring
        except ValueError:
            return ""

    def generate_environment(self, image_path, is_user_feedback=False, feedback_message=""):
        """
        Generates a textual description of an environment based on an input image, using OpenAI's GPT model with vision capabilities.

        Parameters:
            image_path (str): The filesystem path to the image file that will be processed.
            is_user_feedback (bool, optional): Flag to indicate if the message is user feedback. Defaults to False.
            feedback_message (str, optional): Feedback message to be included in the prompt. Defaults to "".

        Returns:
            str: A textual description of the environment as interpreted by the GPT model.
        """
        # Encode the image to base64
        encoded_image = self.encode_image_to_base64(image_path)

        # Load and format the content from the prompt files
        input_environment_content = self.load_and_format_content(
            os.path.join(dir_prompt, 'input_environment.txt')
        )
        prompt_environment_content = self.load_and_format_content(
            os.path.join(dir_prompt, 'prompt_environment.txt'),
        )

        # Combine the content from both files into one prompt content
        if is_user_feedback:
            # Use feedback to modify the prompt environment content
            environment_prompt_content = f"{feedback_message}\n{prompt_environment_content}"
        else:
            environment_prompt_content = input_environment_content + prompt_environment_content

        # Craft the image prompt with text and image content
        image_prompt = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": environment_prompt_content  # Use the combined environment prompt content here
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }

        messages = [image_prompt]

        # Parameters for OpenAI API call
        params = {
            "model": "gpt-4o",
            "messages": messages,
            "headers": {"Openai-Version": "2020-11-07"},
            "max_tokens": 4096,
        }

        # Send the request to GPT
        environment_result = openai.ChatCompletion.create(**params)

        # Process the response
        environment_content = environment_result.choices[0]["message"]["content"]
        print("Received environment description:", environment_content)
        environment = self.extract_json_part(environment_content)

        return environment

    def generate(self, message, environment, image_path, is_user_feedback=False):
        """
        Processes a given instruction or message and interacts with the GPT model to generate a response.
        This method supports handling both new instructions and user feedback on previous responses.
        It dynamically updates the interaction context (messages) based on whether the input is new or feedback and constructs a prompt to send to the GPT model.

        Parameters:
            message (str): The instruction or feedback provided by the user.
            environment (dict): The current environmental context represented as a dictionary. This context includes details about the environment, objects, and their states.
            is_user_feedback (bool, optional): Flag to indicate whether the provided message is feedback on a previous response. Default is False.

        Returns:
            dict: A dictionary representing the LLM response.
        """
        if is_user_feedback:
            self.messages.append({'role': 'user', 'content': message + "\n" + self.instruction})
        else:
            text_base = self.query
            if text_base.find('[ENVIRONMENT]') != -1:
                text_base = text_base.replace('[ENVIRONMENT]', json.dumps(environment))
            if text_base.find('[INSTRUCTION]') != -1:
                text_base = text_base.replace('[INSTRUCTION]', message)
                self.instruction = text_base
            self.messages.append({'role': 'user', 'content': text_base})

        if image_path:
            encoded_image = self.encode_image_to_base64(image_path)
        else:
            encoded_image = None

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=self.create_prompt(encoded_image=encoded_image),
            temperature=0.1,
            max_tokens=self.max_completion_length,
            top_p=0.5,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        text = response['choices'][0].message.content
        print("Received response:", text)
        self.last_response = text
        self.last_response = self.extract_json_part(self.last_response)
        self.last_response = self.last_response.replace("'", "\"")

        # Save the last response
        with open('last_response.txt', 'w') as f:
            f.write(self.last_response)

        try:
            self.json_dict = json.loads(self.last_response, strict=False)
            self.environment = self.json_dict["environment_after"]
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            self.json_dict = {}
        except BaseException:
            self.json_dict = None
            pdb.set_trace()

        if len(self.messages) > 0 and self.last_response is not None:
            self.messages.append({'role': 'assistant', 'content': self.last_response})

        return self.json_dict

    def dump_json(self, dump_name=None):
        """
        Serializes the current state or any structured data into a JSON file.

        Parameters:
            dump_name (Optional[str]): The name of the file to which the JSON data will be written.
        Effect:
            Creates a new JSON file or overwrites an existing file.
        """
        if dump_name is not None:
            # dump the dictionary to a JSON file
            fp = os.path.join('./out', dump_name + '.json')
            with open(fp, 'w') as f:
                json.dump(self.json_dict, f, indent=4)


if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node('ChatGPT_vision_control_node', anonymous=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        help='Scenario name (see the code for details)')
    args = parser.parse_args()
    scenario_name = args.scenario

    # Define the relative path to the image
    image_path = os.path.join('..', 'out', 'demo', 'detected_markers.jpg')

    # Initialize the model and generate the environment description
    aimodel = ChatGPT(credentials, prompt_load_order=prompt_load_order)
    environment_json = aimodel.generate_environment(image_path)

    try:
        environment = json.loads(environment_json)
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)

    # Get user feedback and possibly refine the environment
    while True:
        user_feedback = input('Feedback for environment (return empty if satisfied, type "q" to quit): ')
        if user_feedback.lower() == 'q':
            exit()
        elif user_feedback == '':
            break
        else:
            environment_json = aimodel.generate_environment(image_path, is_user_feedback=True,
                                                            feedback_message=user_feedback)
            environment = json.loads(environment_json)
            print("Refined environment:", json.dumps(environment, indent=4))

    # Proceed to receive and process instructions
    output_dir = os.path.join('./out', scenario_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    instructions = [input("Enter the arm instructions: ")]
    for i, instruction in enumerate(instructions):
        response = aimodel.generate(instruction, environment, image_path, is_user_feedback=False)
        while True:
            user_feedback = input('User feedback (return empty if satisfied, type "q" to quit): ')
            if user_feedback.lower() == 'q':
                exit()
            elif user_feedback == '':
                break
            else:
                response = aimodel.generate(user_feedback, environment, image_path, is_user_feedback=True)

        aimodel.dump_json(f'{scenario_name}/{i}')

    json_path = os.path.join('..', 'out', 'demo', '0.json')

