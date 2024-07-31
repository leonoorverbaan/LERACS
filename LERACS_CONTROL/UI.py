import tkinter as tk
from tkinter import simpledialog, scrolledtext, ttk
from tkinter import *
from PIL import Image, ImageTk
import os
import sys
import json
import rospy
import customtkinter
from PIL import ImageDraw
import subprocess

customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


# Ensure the directory of robot_control.py is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'MRIRAC_control'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GroundingDINO'))

from robot_control import RobotControl
from ChatGPT_init import ChatGPT

with open('secrets.json') as f:
    credentials = json.load(f)

# with open('../secrets.json') as f:
# credentials = json.load(f)

dir_system = './system'
dir_prompt = './/prompt'
dir_query = './/query'
prompt_load_order = ['prompt_role',
                     'input_environment',
                     'prompt_function',
                     'prompt_environment',
                     'prompt_output_format',
                     'prompt_example']

class RobotControlUI(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Robot Control Interface")

        self.geometry(f"{1100}x{1000}")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=4)  # Image row
        self.grid_rowconfigure(1, weight=1)  # Chat area row
        self.grid_rowconfigure(2, weight=0)  # User input row

        self.robot_control = RobotControl()
        self.task_decomposition = ChatGPT(credentials, prompt_load_order)
        self.current_environment = None

        self.style = ttk.Style(self)
        self.create_widgets()

    def create_widgets(self):
        # Create a common font for the labels
        label_font = customtkinter.CTkFont(size=14, weight="bold")
        chat_font = customtkinter.CTkFont(size=13)

        # Create menu and content frames
        self.menu_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0, fg_color="white")
        self.menu_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.menu_frame.grid_rowconfigure(10, weight=1)  # Updated to have more rows
        self.logo_label = customtkinter.CTkLabel(self.menu_frame, text="Franka Interface",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.init_button = customtkinter.CTkButton(self.menu_frame, text="Start Franka",
                                                   command=self.initialize_environment)
        self.init_button.grid(row=1, column=0, padx=20, pady=10)

        self.refresh_button = customtkinter.CTkButton(self.menu_frame, text="Refresh Franka",
                                                      command=self.reinitialize_robot_control)
        self.refresh_button.grid(row=2, column=0, padx=20, pady=10)

        self.detect_button = customtkinter.CTkButton(self.menu_frame, text="Run Detection",
                                                     command=self.run_detection_and_segmentation)
        self.detect_button.grid(row=3, column=0, padx=20, pady=10)

        # Empty row for spacing
        self.menu_frame.grid_rowconfigure(4, minsize=535)

        # Camera Index Option Menu
        self.camera_index_label = customtkinter.CTkLabel(self.menu_frame, text="Camera Index:", anchor="w")
        self.camera_index_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.camera_index_optionemenu = customtkinter.CTkOptionMenu(self.menu_frame,
                                                                    values=["1", "2", "3", "4", "5"],
                                                                    command=self.change_camera_index_event)
        self.camera_index_optionemenu.grid(row=6, column=0, padx=20, pady=(0, 10))
        self.camera_index_optionemenu.set("4")

        self.appearance_mode_label = customtkinter.CTkLabel(self.menu_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.menu_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 10))

        self.scaling_label = customtkinter.CTkLabel(self.menu_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.menu_frame,
                                                               values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=10, column=0, padx=20, pady=(10, 20))

        self.content_frame = customtkinter.CTkFrame(self, width=750, height=700, fg_color="white")
        self.content_frame.grid(row=0, column=1, rowspan=3, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)  # Label row
        self.content_frame.grid_rowconfigure(1, weight=3)  # Image row
        self.content_frame.grid_rowconfigure(2, weight=1)  # Chat area row
        self.content_frame.grid_rowconfigure(3, weight=1)  # User input row

        # Create a frame around the "Robot Vision" label with less sharp corners
        self.robot_vision_label_frame = customtkinter.CTkFrame(self.content_frame, corner_radius=10, fg_color="#DAE9F8")
        self.robot_vision_label_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.robot_vision_label_frame.grid_columnconfigure(0, weight=1)

        # Add the "Robot Vision" label inside the frame
        self.robot_vision_label = customtkinter.CTkLabel(self.robot_vision_label_frame, corner_radius=10, text="Robot Vision",
                                                         font=label_font, fg_color="#6EA8E5")
        self.robot_vision_label.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Create a dark grey frame around the image
        self.image_frame = customtkinter.CTkFrame(self.content_frame, corner_radius=10, width=750, fg_color="#DAE9F8")
        self.image_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        # Updated to place image inside the new frame
        self.img_label = customtkinter.CTkLabel(self.image_frame, text="", fg_color="#DAE9F8")  # Make label text empty
        self.img_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.img_label_detected = customtkinter.CTkLabel(self.image_frame, text="", fg_color="#DAE9F8")  # Make label text empty
        self.img_label_detected.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Create a scrollable frame for chat area
        self.scrollable_frame = customtkinter.CTkScrollableFrame(self.content_frame, label_text="Chat Area", width=750,
                                                                 height=10, label_font=label_font, fg_color="#DAE9F8")
        self.scrollable_frame.grid(row=2, column=0, pady=(10, 10), sticky="nsew")
        self.scrollable_frame._label.configure(fg_color="#6EA8E5")

        # Add CTkTextbox to the scrollable frame
        self.chat_area = customtkinter.CTkTextbox(self.scrollable_frame, wrap=tk.WORD, font=chat_font)
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        self.chat_area.configure(state=tk.DISABLED)

        self.input_frame = customtkinter.CTkFrame(self.content_frame, height=40, corner_radius=10, fg_color="#DAE9F8")
        self.input_frame.grid(row=3, column=0, pady=(10, 10), padx=10, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        # Entry widget for user input
        self.user_input = ttk.Entry(self.input_frame, width=100)
        self.user_input.grid(row=0, column=0, padx=15, pady=15, sticky="ew")
        self.user_input.bind("<Return>", self.process_environment_feedback)

        self.bind("<Escape>", self.close_window)
        self.image_path = os.path.join(os.path.dirname(__file__), 'MRIRAC_control', 'out', 'demo',
                                       'detected_markers.jpg')

        self.loading_label = customtkinter.CTkLabel(self, text="Loading...", font=label_font, text_color="#6EA8E5")
        self.loading_label.grid(row=0, column=1, padx=20, pady=10)
        self.loading_label.grid_remove()


    def display_image(self, image_path, label):
        """Display an image from a given path with rounded corners."""
        image = Image.open(image_path)
        image = image.resize((400, 262), Image.LANCZOS)  # Smaller size for side-by-side display

        # Create rounded corner mask
        corner_radius = 30  # Adjust this value as needed
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([0, 0, image.size[0], image.size[1]], corner_radius, fill=255)

        # Apply mask to the image
        rounded_image = Image.new("RGBA", image.size)
        rounded_image.paste(image, (0, 0), mask=mask)

        # Convert to ImageTk
        photo = ImageTk.PhotoImage(rounded_image)
        label.configure(image=photo)
        label.image = photo

    def reinitialize_robot_control(self):
        """Reinitialize the robot control node."""
        self.robot_control.camera.close()  # Close the camera
        del self.robot_control  # Delete the existing instance

        # Create a new instance of RobotControl
        self.robot_control = RobotControl()

        rospy.sleep(1)  # Allow some time for the new instance to initialize
        self.append_to_chat("Robot control node reinitialized.", sender="robot")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def change_camera_index_event(self, new_camera_index: str):
        camera_index = int(new_camera_index)
        self.robot_control.update_camera_index(camera_index)

    def append_to_chat(self, message, sender="robot"):
        self.chat_area.configure(state=tk.NORMAL)  # Use 'configure' instead of 'config'

        # Apply formatting based on the sender
        tag_name = 'robot' if sender == 'robot' else 'you'
        self.chat_area.tag_config(tag_name, foreground="dark blue" if sender == 'robot' else "dark blue")

        # Insert the sender label
        sender_label = f"{sender.capitalize()}: "
        self.chat_area.insert(tk.END, sender_label, tag_name)

        # Insert the message
        if isinstance(message, dict):
            formatted_message = json.dumps(message, indent=4)
        else:
            formatted_message = message
        self.chat_area.insert(tk.END, formatted_message + "\n\n")

        self.chat_area.configure(state=tk.DISABLED)  # Use 'configure' instead of 'config'
        self.chat_area.yview(tk.END)

    def initialize_environment(self):
        """Initialize the environment and update constraints."""
        self.robot_control.all_update()
        self.robot_control.init_constraints()
        self.display_image(self.image_path, self.img_label)

        aimodel = ChatGPT(credentials, prompt_load_order=prompt_load_order)

        try:
            environment_json = aimodel.generate_environment(self.image_path)
            self.current_environment = json.loads(environment_json)
            self.append_to_chat(json.dumps(self.current_environment, indent=4), sender="robot")
            self.append_to_chat("Feedback for environment (return empty if satisfied, type 'q' to quit):",
                                sender="robot")
        except json.JSONDecodeError as e:
            self.append_to_chat("Failed to decode environment data. Please try again.", sender="robot")

    def process_environment_feedback(self, event=None):
        feedback = self.user_input.get().strip()
        self.append_to_chat(feedback, sender="you")
        self.user_input.delete(0, tk.END)

        if feedback == '':
            self.append_to_chat("Environment approved. Enter task instructions.", "robot")
            self.user_input.bind("<Return>", self.process_instruction_feedback)
        elif feedback.lower() == 'q':
            self.append_to_chat("Operation terminated by user.", "robot")
            self.destroy()
        else:
            aimodel = ChatGPT(credentials, prompt_load_order=prompt_load_order)
            try:
                environment_json = aimodel.generate_environment(self.image_path, is_user_feedback=True, feedback_message=feedback)
                self.current_environment = json.loads(environment_json)
                self.append_to_chat(self.current_environment, "robot")
                self.append_to_chat("Provide feedback for the environment, or press Enter to approve.", "robot")
            except json.JSONDecodeError as e:
                self.append_to_chat("Failed to decode JSON. Please try again.", "robot")

    def process_instruction_feedback(self, event=None):
        user_input = self.user_input.get().strip()
        self.append_to_chat(self.user_input.get().strip(), sender="you")
        self.user_input.delete(0, tk.END)

        # Initialize the output directory only once
        if not hasattr(self, 'output_dir'):
            scenario_name = "demo"  # Use a fixed scenario name or modify as needed
            self.output_dir = os.path.join('./out', scenario_name)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        if user_input == '':  # Check if the user input is empty and response exists
            if hasattr(self, 'response'):  # Check if there's a response to save and execute
                json_path = os.path.join(self.output_dir, '0.json')
                with open(json_path, 'w') as f:
                    json.dump(self.response, f, indent=4)
                self.append_to_chat(f"Task sequence saved and ready to execute: {json_path}", "robot")
                self.robot_control.run_task_sequence(json_path)
                del self.response
            return

        # Handle the input as either new instruction or feedback
        if not hasattr(self, 'response'):
            try:
                self.response = self.task_decomposition.generate(user_input, self.current_environment, self.image_path,
                                                                 is_user_feedback=False)
                self.append_to_chat(self.response, "robot")
                self.append_to_chat("Provide feedback for the instructions, or press Enter to approve.", "robot")
            except Exception as e:
                self.append_to_chat("Failed to generate initial response: " + str(e), "robot")
        else:  # Handle user input as feedback to modify the existing response
            try:
                self.response = self.task_decomposition.generate(user_input, self.current_environment, self.image_path,
                                                                 is_user_feedback=True)
                self.append_to_chat(self.response, "robot")
                self.append_to_chat("Provide feedback for the instructions, or press Enter to approve.", "robot")
            except Exception as e:
                self.append_to_chat("Failed to update response: " + str(e), "robot")

    def run_detection_and_segmentation(self):
        """Run detection and segmentation as a subprocess and display the result."""
        output_directory = '/home/allianderai/LLM-franka-control-rt/src/rcdt_LLM_fr3/MRIRAC_control/out/demo'
        json_file_path = os.path.join(output_directory, "0.json")
        script_path = "/home/allianderai/LLM-franka-control-rt/src/GroundingDINO/detection_and_segmentation.py"

        command = [
            "python", script_path,
            "--json_file_path", json_file_path,
            "--source_image_path", self.image_path
        ]

        # Execute the script as a subprocess
        subprocess.run(command)

        # Display the output image next to the original image
        detected_image_path = os.path.join(output_directory, "segmented_annotated_image_20240520_151203.jpg")
        self.display_image(detected_image_path, self.img_label_detected)

    def close_window(self, event=None):
        """Close the window."""
        self.destroy()
        

if __name__ == "__main__":
    rospy.init_node("robot_control_ui_node", anonymous=True)
    app = RobotControlUI()
    app.mainloop()
