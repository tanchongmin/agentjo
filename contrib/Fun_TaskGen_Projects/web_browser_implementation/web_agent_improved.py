import datetime
import os

import pyautogui
from helium import Link, find_all, go_to, kill_browser, start_chrome, wait_until
from scripts.env_loader import load_api_key
from selenium.common.exceptions import NoSuchElementException

from taskgen import Agent, Function

api_key = load_api_key()


def stop_web_browser():
    kill_browser()
    return {"Output": "Web browser stopped"}


def start_browser():
    start_chrome()
    return {"Output": "Web browser started."}


def perform_search(search_query: str):
    go_to("https://www.startpage.com")
    pyautogui.typewrite(search_query)
    pyautogui.press("enter")
    wait_until(lambda: len(find_all(Link())) > 0, timeout_secs=10)
    return {"Output": "Search performed"}


def navigate_to_url(url: str):
    go_to(url)
    return {"Output": f"Navigated to {url}"}


def get_all_links():
    # Create links folder in the current directory
    links_dir = os.path.join(os.getcwd(), "links")
    if not os.path.exists(links_dir):
        os.makedirs(links_dir)

    # Wait until at least one link is visible or a certain timeout is reached
    wait_until(lambda: len(find_all(Link())) > 0, timeout_secs=10)

    links = find_all(Link())
    # Store it in links folder
    links_file_path = os.path.join(
        links_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    )
    with open(links_file_path, "w") as f:
        for link in links:
            try:
                href = link.web_element.get_attribute("href")
                if href:  # Check if href is not None
                    f.write(href + "\n")
            except NoSuchElementException:
                continue  # Skip this link if no href attribute is found

    return {"Output": f"Links stored in {links_file_path}"}


# def click_first_link():
#     # Wait until element is visible
#     wait_until(Link("Images").exists)

#     links = find_all(Link())
#     print(f"Number of links found: {len(links)}")  # Debugging output

#     if links:
#         first_link = links[0]
#         print(
#             f"Attempting to click on the first link: {first_link}"
#         )  # Debugging output
#         click(first_link)
#         return {"Output": "Clicked the first link"}
#     else:
#         return {"Output": "No links found on the page"}


# def fill_form(form_details: dict):
#     for field, value in form_details.items():
#         pyautogui.typewrite(value)
#         pyautogui.press("tab")
#     pyautogui.press("enter")  # To submit the form
#     return {"Output": "Form filled"}
# def interact_with_element(element_identifier, action="click"):
#     element = find(element_identifier)
#     if action == "click":
#         click(element)
#     elif action == "hover":
#         pyautogui.moveTo(element.center_x, element.center_y)
#     return {"Output": f"{action.capitalize()} on element"}
# def scroll_page(direction="down", amount=3):
#     if direction == "down":
#         pyautogui.scroll(-amount)
#     else:
#         pyautogui.scroll(amount)
#     return {"Output": f"Scrolled {direction}"}
# def capture_screenshot():
#     screenshot = pyautogui.screenshot()
#     screenshot.save("screenshot.png")
#     return {"Output": "Screenshot captured"}
# def handle_alert(accept=True):


# Define the functions
fn_list = [
    Function(
        fn_description="Stop the web browser",
        external_fn=stop_web_browser,
        output_format={"Output": "Web browser stopped"},
    ),
    Function(
        fn_description="Opens web browser",
        external_fn=start_browser,
        output_format={"Output": "Web browser started"},
    ),
    Function(
        fn_description="Navigate to a URL <url: str>",
        external_fn=navigate_to_url,
        output_format={"Output": "Navigated to URL"},
    ),
    Function(
        fn_description="Perform a search <search_query: str>",
        external_fn=perform_search,
        output_format={"Output": "Search performed"},
    ),
    Function(
        fn_description="Get all links on the page",
        external_fn=get_all_links,
        output_format={"Output": "Links stored in a file"},
    ),
]

# Create the agent
agent = Agent(
    "Web Browser Agent",
    "Performs web browser actions.",
    shared_variables={"driver": ""},
).assign_functions(fn_list)

# See the auto-generated names of your functions
agent.list_functions()

# Run the agent
# output = agent.run(
#     "Open browser, search for cats, go to google.com, get all links , stop browser."
# )
output = agent.run("Open browser, search for cats, get all links , stop browser.")
print(output)
