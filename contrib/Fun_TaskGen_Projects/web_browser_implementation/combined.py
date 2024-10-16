import os
from datetime import datetime

import pyautogui
from scripts.actions import (
    Create_a_new_instance_of_the_Chrome_driver,
    Navigate_to_a_web_page,
)
from utils.create_screenshot_folder import create_screenshots_folder
from utils.env_loader import load_api_key

from taskgen import Agent, Function

# Load API key
api_key = load_api_key()

# LLM-based function example
llm_function = Function(
    fn_description="Generate a greeting message for <name>",
    output_format={"output": "greeting message"},
)


def type_text(text: str) -> str:
    pyautogui.typewrite(text)
    return f"Text typed: {text}"


def click_screen(x: int, y: int) -> str:
    pyautogui.click(x, y)
    return f"Action status: clicked at coordinates ({x}, {y})"


def take_screenshot(shared_variables, x: int):
    screenshots_folder = create_screenshots_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_filename = f"screenshot_{timestamp}.png"
    screenshot_path = os.path.join(screenshots_folder, screenshot_filename)
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)
    return {"Output": f"Screenshot saved at: {screenshot_path}"}


def create_writer_agent():
    fn_list = [
        Function(
            fn_description="Type text <text: str>",
            external_fn=type_text,
            output_format={"Output": "Text typed"},
        ),
        Function(
            fn_description="Click the screen at coordinates <x: int> and <y: int>",
            external_fn=click_screen,
            output_format={"Output": "Action status"},
        ),
    ]
    agent = Agent(
        "Writer Agent",
        "Performs text typing and screen interaction actions",
        verbose=True,
    ).assign_functions(fn_list)
    return agent


def create_web_agent():
    fn_list = [
        Function(
            fn_description="Create a new instance of the Chrome driver <x: int>",
            external_fn=Create_a_new_instance_of_the_Chrome_driver,
            output_format={"Output": "Chrome driver instance"},
        ),
        Function(
            fn_description="Navigate to a web page <url: str>",
            external_fn=Navigate_to_a_web_page,
            output_format={"Output": "Navigation status"},
        ),
    ]
    agent = Agent(
        "Screen Interaction Agent",
        "Performs screen interaction actions",
        shared_variables={"Driver": ""},
    ).assign_functions(fn_list)
    return agent


# Base Agent
def create_base_agent():
    base_agent = Agent(
        "Base Agent",
        "Orchestrates tasks using the writer, manager and web agents",
        verbose=True,
    )
    return base_agent


if __name__ == "__main__":
    base_agent = create_base_agent()

    # Example task for the base agent
    task = "click the screen at coordinates 1000 and 200, then type the text 'Hello, world!', then take a screenshot."

    # Create instances of the writer and manager agents
    writer_agent = create_writer_agent()
    web_agent = create_web_agent()

    # Assign the web and actions agents to the base agent
    base_agent.assign_agents([writer_agent, web_agent])

    # Run the base agent to complete the task
    output = base_agent.run(task)
    print(f"Base Agent output: {output}")
