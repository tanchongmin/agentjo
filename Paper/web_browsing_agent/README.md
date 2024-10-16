# Web Browser Implementation
This example uses the Taskgen framework to create a web browser agent that can interact with web pages.     
The agent can take screenshots, extract text, and interact with interactive elements on the page.

## Introduction

## Features
The agent will be able to perform the following tasks:
- Open a web page
- Going to a specific URL
- Extract relevant content from the page
- Saves context in context.txt file

For more details, refer to the tasks list:

## Prerequisites
- Python 3.10
- taskgen library
- python-dotenv library
- OpenAI API key

## Setup
Create a `.env` file in the project root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

Install the required dependencies:
```bash
# Create a venv file in the root project directory
python -m venv venv

# Activate the venv environment
source venv/bin/activate

# Install the required dependencies
cd Paper/web_browsing_agent && pip install -r requirements.txt

# Run the web browsing agent
python conversational_web_browsing_agent.py
```

## Tasks given:

| Queries Completed | Task Description |
|-------------------|------------------|
| 0/5 | Search 'impact of social media on mental health' and summarize the academic studies. |
| 5/5 | Visit 'https://www.who.int' and summarize the latest health advisories. |
| 5/5 | Open the browser, search 'weather forecast New York', and save the first result. |
| 5/5 | Navigate to 'https://www.bbc.com/news', extract the top news headlines, and summarize them. |
| 0/5 | Search 'global warming statistics 2024' and provide a summary of the data trends. |
| 5/5 | Visit 'https://www.finance.yahoo.com', gather the latest stock market updates, and summarize. |
| 5/5 | Search 'Shakespeare's influence on modern literature' and summarize the academic articles. |
| 5/5 | Search 'quantum computing vs classical computing' and summarize the differences from multiple sources. |
| 5/5 | Visit 'https://www.nasa.gov', gather the latest Mars mission updates, and cross-reference with Wikipedia. |
| 1/5 | Search 'key events in AI development 2024' and summarize the timeline. |
| 3/5 | Search 'market analysis of electric vehicles 2024' and summarize the findings. |
| 5/5 | Search 'best noise-canceling headphones 2024' and summarize the top reviews. |
| 5/5 | Navigate to 'https://www.consumerreports.org', gather information on washing machines, and summarize the best options. |
| 5/5 | Visit 'https://docs.python.org', find information on Python decorators, and summarize. |
| 2/5 | Search 'workplace safety measures during COVID-19' and summarize the guidelines. |
| 4/5 | Visit 'https://www.cdc.gov', find information on flu prevention, and summarize. |
| 0/5 | Search 'latest trends in renewable energy 2024' and summarize the key developments. |
| 5/5 | Visit 'https://www.techcrunch.com', gather the latest technology news, and summarize. |
| 4/5 | Search 'evolution of jazz music' and summarize its impact on modern genres. |
| 5/5 | Visit 'https://www.metmuseum.org', explore the latest exhibits, and summarize. |

## User guide:
1. Run `python conversational_web_browsing_agent.py`
2. Users are able to ask questions on the search after completion from agent's memory.
3. When performing a new search, the agent should be exited by typing 'exit' and then re-entering the search query.
4. If having problems with reliability start with "Open the browser" followed by "search 'query'" and action to be taken.