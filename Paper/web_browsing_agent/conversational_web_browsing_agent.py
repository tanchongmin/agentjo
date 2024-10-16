import os
from typing import Tuple

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from helium import get_driver, go_to, kill_browser, start_chrome
from openai import OpenAI
from selenium.webdriver.common.keys import Keys

from taskgen import Agent


# Load the API key from the .env file
def load_api_key(dotenv_path="../../.env"):
    load_dotenv(dotenv_path)
    return os.getenv("OPENAI_API_KEY")


load_api_key()


# Define the functions for browser control
def open_browser() -> str:
    """Opens the web browser using Helium"""
    start_chrome()
    return {"Output": "Web browser opened. Currently on empty page."}


def close_browser() -> str:
    """Closes the web browser using Helium"""
    kill_browser()
    return {"Output": "Web browser closed."}


def _browser_state() -> Tuple[str, str]:
    driver = get_driver()
    header = f"Address: {driver.current_url}\n"
    header += f"Title: {driver.title}\n"
    content = driver.page_source
    return (header, content)


def clean_html(content: str) -> str:
    soup = BeautifulSoup(content, "lxml")
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator="\n")


def extract_relevant_content(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "lxml")

    # Remove unwanted tags
    for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
        script.decompose()

    # Extract headings and paragraphs
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    paragraphs = soup.find_all("p")

    # Combine the text from headings and paragraphs
    content = "\n".join(
        [heading.get_text() for heading in headings]
        + [para.get_text() for para in paragraphs]
    )

    # Optionally, limit the length of the content
    max_length = 2000  # Adjust as needed
    if len(content) > max_length:
        content = content[:max_length] + "..."

    return content


def save_context_to_file(header: str, content: str, filename: str = "context.txt"):
    clean_content = extract_relevant_content(content)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(header + "\n=======================\n" + clean_content)


def informational_search(query: str) -> str:
    go_to(f"https://www.bing.com/search?q={query}")
    header, content = _browser_state()
    save_context_to_file(header, content)
    return {
        "Output": f"Performed informational search for '{query}' and saved context."
    }


def navigational_search(query: str) -> str:
    go_to(f"https://www.bing.com/search?q={query}")
    driver = get_driver()
    first_result = driver.find_element_by_css_selector("a")
    first_result.send_keys(Keys.RETURN)
    header, content = _browser_state()
    save_context_to_file(header, content)
    return {"Output": f"Performed navigational search for '{query}' and saved context."}


def visit_page(url: str) -> str:
    go_to(url)
    header, content = _browser_state()
    save_context_to_file(header, content)
    return {"Output": f"Visited page at {url} and saved context."}


def page_up() -> str:
    driver = get_driver()
    driver.execute_script("window.scrollBy(0, -window.innerHeight);")
    header, content = _browser_state()
    save_context_to_file(header, content)
    return {"Output": "Scrolled up and saved context."}


def page_down() -> str:
    driver = get_driver()
    driver.execute_script("window.scrollBy(0, window.innerHeight);")
    header, content = _browser_state()
    save_context_to_file(header, content)
    return {"Output": "Scrolled down and saved context."}


def summarize_context(filename: str = "context.txt") -> str:
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to summarize web pages.",
            },
            {
                "role": "user",
                "content": "Please summarize the following content:\n\n" + content,
            },
        ],
        max_tokens=150,
    )

    summary = response.choices[0].message.content.strip()
    return {"Output": f"Summary: {summary}"}


# Define the functions for web surfing
fn_list_3 = [
    informational_search,
    navigational_search,
    visit_page,
    open_browser,
    close_browser,
    summarize_context,
]

WebSurfer = Agent(
    "WebSurfer",
    "Performs web searches and navigates web pages. Always open the browser at the start of the task and close the browser at the end.",
    model="gpt-4o",
    default_to_llm=False,
).assign_functions(fn_list_3)

# Define the boss agent (meta agent) that controls other agents
bossagent = Agent(
    "WebNavigator",
    "Assists user to navigate the web. Always open the browser at the start of the task and close the browser at the end.",
    model="gpt-4o",
    default_to_llm=False,
)

# Update the boss agent to include the new WebSurfer agent
agent_list = [WebSurfer]
bossagent.assign_agents(agent_list)


# Define the help function
def display_help():
    help_text = """
Available commands:
  - open_browser: Opens the web browser
  - close_browser: Closes the web browser
  - type_text <text>: Types the given text
  - enter_key: Presses the Enter key
  - navigate_to_url_via_address_bar <url>: Navigates to the specified URL
  - exit: Exits the conversation
  - help: Displays this help message
"""
    print(help_text)


def main():
    print("Welcome to the WebNavigator CLI!")
    print("Type 'help' for a list of commands or 'exit' to quit.")
    print("Note: This CLI uses Bing for search queries.")
    print("Example query: Open browser, Search 'hello world !' and summarise the content.")

    while True:
        query = input("User: ")
        if query.lower() == "exit":
            print("Exiting the conversation.")
            break
        elif query.lower() == "help":
            display_help()
        else:
            output = bossagent.run(query)
            print(output)


if __name__ == "__main__":
    main()
