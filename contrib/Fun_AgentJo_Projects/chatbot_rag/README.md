# Conversational Agent
This script demonstrates a simple conversational agent that can engage in personalized conversations with users. The agent is built using the taskgen library and utilizes OpenAI's API for natural language processing.

## Features
- Greets the user and stores their name in memory
- Ingests JSON data from a file and stores it in memory
- Retrieves relevant documents based on user queries
- Supports custom functions for generating sentences with specific words and styles
## Prerequisites
- Python 3.10
- taskgen library
- python-dotenv library
- OpenAI API key
## Setup
Install the required dependencies:

```bash
pip install -r requirements.txt
```
Create a `.env` file in the project root directory and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```
## Data Ingestion
Prepare a JSON file containing the documents you want to ingest. The JSON file should have the following structure:
```json
[
  {
    "name": "Document 1",
    "content": "Content of document 1"
  },
  {
    "name": "Document 2",
    "content": "Content of document 2"
  }
]
```
Place the JSON file in the `data` directory with the name `documents.json`.

## Usage
1. Enter your name when prompted by the agent.

2. The agent will greet you and store your name in memory.

3. The agent will ingest the JSON data from data/documents.json and store it in memory.

4. Enter a query to retrieve relevant documents based on your query.

5. The agent will display the names of the most relevant documents for your query.

## Customization
You can modify the fn_list to add or remove custom functions for the conversational agent.
Update the path variable to specify a different JSON file for ingesting documents.
Customize the output_format of the functions to change the structure of the agent's responses.