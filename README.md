# AgentJo v1.0.0
### Towards Human-Friendly, Fast Learning and Adaptable Agent Communities
#### An Open Source Initiative Led by [John Tan Chong Min](https://www.linkedin.com/in/john-chong-min-tan-94652288/)
#### This is John's dream to create fast learning and adaptable agents
<p align="center">
   <img src="./logo/agentjo_logo.png" alt="AgentJo Logo" width="400">
</p>
 
Road Map (Currently at Alpha):
![AgentJo Future Plans](./resources/Future_Plans.png)

Agentic System Overview:
![AgentJo Overview](./resources/TaskGen_Overview.png)

- Discussion Channel + AgentJo Logo Design Competition (John's AI Group): [https://discord.gg/bzp87AHJy5](https://discord.gg/bzp87AHJy5)
- Grand Vision of AgentJo: https://www.youtube.com/watch?v=u1BHvKRnhYY

- AgentJo v0 (TaskGen) Paper: https://web3.arxiv.org/pdf/2407.15734
- AgentJo v0 (TaskGen) Video: https://www.youtube.com/watch?v=F3usuxs2p1Y
- Related Repositories: StrictJSON (https://github.com/tanchongmin/strictjson) [Do help star this as well!]

### Acknowledgements
- Idea for AgentJo was built over 5 years (2019 - 2024) during John's PhD, guided by Prof. Mehul Motani
- Initial version (v0) of AgentJo (TaskGen) was done during John's time at Simbian AI (Feb - Oct 2024), alongside talented individuals such as Prince Saroj, Brian Lim, Richard Cottrill, Bharat Runwal, Hardik
- API key sponsorship for AgentJo's development (v1 and above) is currently sponsored by:
    - Jostin Pte. Ltd. and Kent Ridge AI
- A shoutout here too for our past supporters sponsoring API keys for development of AgentJo v0: Yanok, Ajentik AI, Simbian AI

### Creator's Preamble
Happy to share that we are beginning the phase to augment agents and incorporate them in larger systems.

What has been done:
- Splitting of Tasks into subtasks for bite-sized solutions for each subtask
- Single Agent with LLM Functions
- Single Agent with External Functions
- Meta Agent with Inner Agents as Functions
- Shared Variables for multi-modality support
- Retrieval Augmented Generation (RAG) over Function space
- Memory to provide additional task-based prompts for task
- Global Context for configuring your own prompts + add persistent variables
- Async mode for Agent, Function and `strict_json`, `parse_yaml`

I am quite sure that this is the best open-source agentic framework for task-based execution out there! 
Existing frameworks like AutoGen rely too much on conversational text which is lengthy and not targeted.
AgentJo uses StrictJSON (LLM structured output parser with type checking and more!) as the core, and agents are efficient and are able to do Chain of Thought natively using structured output.

What can you do to help (see contrib folder for more details): 
- Star the github so more people can use it (It's open source and free to use, even commercially!)
- Contribute template Jupyter Notebooks for your favourite use cases so it can be much more boilerplate for others to use :)
- Contribute Agent Wrappers that imbue additional functions to the base agent (e.g Planner, Reflector, Conversation)
- Contribute Memory classes that allow the Agent to store and extract various kinds of memories according to the task

I can't wait to see what this new framework can do for you!

### Benefits of structured output over agentic frameworks using conversational free-text like AutoGen
- Helps do Chain-of-Thought prompting naturally and is less verbose than free text
- Allows for natural parsing of multiple output fields by agents
- `strict_json` / `parse_yaml` helps to ensure all output fields are there and of the right format required for downstream processing

### Creator Info
- Created: 17 Feb 2024 by [John Tan Chong Min](https://www.linkedin.com/in/john-chong-min-tan-94652288/)
- Lead Documentation: [Brian Lim](https://www.linkedin.com/in/brianlimyisheng/)
- Paper Research Staff: [Prince Saroj](https://www.linkedin.com/in/psaroj/), [Hardik Maheshwari](https://www.linkedin.com/in/hardik1496/), [Bharat Runwal](https://www.linkedin.com/in/bharat-runwal-673144196/), [Brian Lim](https://www.linkedin.com/in/brianlimyisheng/), [Richard Cottrill](https://www.linkedin.com/in/richardc/)
- Logo Designer: [Boddu Sri Pavan](https://www.linkedin.com/in/boddusripavan/)
- Collaborators welcome

## How do I use this? 
1. Download package via command line ```pip install agentjo```
(For file reading capabilities in memory, use ```pip install agentjo[full]```)
3. Set up your LLM and provide any API keys if needed
4. Import the required functions from ```agentjo``` and use them!

## Recommendation for LLM for Agentic Framework
- Default model is now gpt-4o-mini
- Thinking models like gpt-o3-mini, gemini-2.0-flash-thinking-exp can be used as well, but many times just adding a "Thoughts" in the `output_format` will suffice and is quicker to process

# 1. Agent Basics
- Create an agent by entering your agent's name and description
- Agents are task-based, so they will help generate subtasks to fulfil your main task

- Agents are made to be non-verbose, so they will just focus only on task instruction (Much more efficient compared to conversational-based agentic frameworks like AutoGen)
- Agent's interactions will be stored into `subtasks_completed` by default, which will serve as a memory buffer for future interactions

- **Inputs for Agent**:
    - **agent_name**: String. Name of agent, hinting at what the agent does
    - **agent_description**: String. Short description of what the agent does
    - **max_subtasks**: Int. Default: 5. The maximum number of subtasks the agent can have
    - **verbose**: Bool. Default: True. Whether to print out agent's intermediate thoughts
    - **llm**: Function. The LLM to be used by the Agent
<br/><br/>

- **Agent Internal Parameters**:
    - **Task**: String. The task the agent has been assigned to - Defaults to "No task assigned"
    - **Subtasks Completed**: Dict. The keys are the subtask names and the values are the result of the respective subtask
    - **Is Task Completed**: Bool. Whether the current Task is completed
<br/><br/>

- **Task Running**
    - **reset()**: Resets the Agent Internal Parameters and Subtasks Completed. You should do this at the start of every new task assigned to the Agent to minimise potential confusion of what has been done for this task versus previous tasks
    - **run(task: str, num_subtasks: int = max_subtasks)**: Performs the task. Do note that agent's state will not be reset, so if you want to reset it, call reset() prior to running this. Runs the task for **num_subtasks** steps. If not specified, we will take the **max_subtasks**.
<br/><br/>

- **Give User Output**
    - **reply_user(query: str = '', stateful: bool = True)**: Using all information from subtasks, give a reply about the `query` to the user. If `query` is not given, then it replies based on the current task the agent is doing. If `stateful` is True, saves this query and reply into `subtasks_completed`
<br/><br/>
    
- **Check status of Agent**:
    - **status()**: Lists out Agent Name, Agent Description, Available Functions (default function is to use the LLM), Task, Subtasks Completed and Is Task Completed
    
## Example Agent Creation
```python
my_agent = Agent('Helpful assistant', 'You are a generalist agent', llm = llm)
```

## Example Agent Task Running - Split the assigned task into subtasks and execute each of them

```python
output = my_agent.run('Give me 5 words rhyming with cool, and make a 4-sentence poem using them')
```

`Subtask identified: Find 5 words that rhyme with 'cool'`

`Getting LLM to perform the following task: Find 5 words that rhyme with 'cool'`
> pool, rule, fool, tool, school

`Subtask identified: Compose a 4-sentence poem using the words 'pool', 'rule', 'fool', 'tool', and 'school'`

`Getting LLM to perform the following task: Compose a 4-sentence poem using the words 'pool', 'rule', 'fool', 'tool', and 'school'`
> In the school, the golden rule is to never be a fool. Use your mind as a tool, and always follow the pool.

`Task completed successfully!`

## Check Agent's Status
```python
my_agent.status()
```

`Agent Name: Helpful assistant`

`Agent Description: You are a generalist agent`

`Available Functions: ['use_llm', 'end_task']`

`Task: Give me 5 words rhyming with cool, and make a 4-sentence poem using them`

`Subtasks Completed:`

`Subtask: Find 5 words that rhyme with 'cool'`

`pool, rule, fool, tool, school`

`Subtask: Compose a 4-sentence poem using the words 'pool', 'rule', 'fool', 'tool', and 'school'`

`In the school, the golden rule is to never be a fool. Use your mind as a tool, and always follow the pool.`

`Is Task Completed: True`

## Example Agent Reply to User - Reference the subtasks' output to answer the user's query
```python
output = my_agent.reply_user()
```

`
Here are 5 words that rhyme with "cool": pool, rule, fool, tool, school. Here is a 4-sentence poem using these words: "In the school, the golden rule is to never be a fool. Use your mind as a tool, and always follow the pool."
`

# 2. Power Up your Agents - Bring in Functions (aka Tools)
- First define the functions, either using class `Function` (see Tutorial 0), or just any Python function with input and output types defined in the signature and with a docstring
- After creating your agent, use `assign_functions` to assign a list of functions of class `Function`, or general Python functions (which will be converted to AsyncFunction)
- Function names will be automatically inferred if not specified
- Proceed to run tasks by using `run()`

```python
# This is an example of an LLM-based function
sentence_style = Function(fn_description = 'Output a sentence with <number> and <entity> in the style of <emotion>', 
                         output_format = {'output': 'sentence'},
                         fn_name = 'sentence_with_number_entities_emotion',
                         llm = llm)

# This is an example of an external user-defined function
def binary_to_decimal(binary_number: str) -> int:
    '''Converts binary_number to integer of base 10'''
    return int(str(binary_number), 2)

# Initialise your agent and assign the functions
my_agent = Agent('Helpful assistant', 'You are a generalist agent', 
        llm = llm).assign_functions([sentence_style, binary_to_decimal])

# Run the Agent
output = my_agent.run('First convert binary string 1001 to a number, then generate me a happy sentence with that number and a ball')
```

`Subtask identified: Convert the binary number 1001 to decimal`
`Calling function binary_to_decimal with parameters {'x': '1001'}`

> {'output1': 9}

`Subtask identified: Generate a happy sentence with the decimal number and a ball`
`Calling function sentence_with_number_entities_emotion with parameters {'number': '9', 'entity': 'ball', 'emotion': 'happy'}`

> {'output': 'I am so happy with my 9 balls.'}

`Task completed successfully!`

- Approach 1: Automatically Run your agent using `run()`

- Approach 2: Manually select and use functions for your task
    - **select_function(task: str)**: Based on the task, output the next function name and input parameters
    - **use_function(function_name: str, function_params: dict, subtask: str = '', stateful: bool = True)**: Uses the function named `function_name` with `function_params`. `stateful` controls whether the output of this function will be saved to `subtasks_completed` under the key of `subtask`
<br/><br/>

- **Assign/Remove Functions**:
    - **assign_functions(function_list: list)**: Assigns a list of functions to the agent
    - **remove_function(function_name: str)**: Removes function named function_name from the list of assigned functions
<br/><br/>

- **Show Functions**:
    - **list_functions()**: Returns the list of functions of the agent
    - **print_functions()**: Prints the list of functions of the agent
<br/><br/>

# 3. AsyncAgent

- `AsyncAgent` works the same way as `Agent`, only much faster due to parallelisation of tasks
- It can only be assigned functions of class `AsyncFunction`, or general Python functions (which will be converted to AsyncFunction)
- If you define your own `AsyncFunction`, you should define the fn_name as well if it is not an External Function
- As a rule of thumb, just add the `await` keyword to any function that you run with the `AsyncAgent`

#### Example LLM in Async Mode
```python
async def llm_async(system_prompt: str, user_prompt: str):
    ''' Here, we use OpenAI for illustration, you can change it to your own LLM '''
    # ensure your LLM imports are all within this function
    from openai import AsyncOpenAI
    
    # define your own LLM here
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model='gpt-4o-mini',
        temperature = 0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content
```

#### Example Agentic Workflow
```python
# This is an example of an LLM-based function
sentence_style = AsyncFunction(fn_description = 'Output a sentence with <number> and <entity> in the style of <emotion>', 
                     output_format = {'output': 'sentence'}, 
                     fn_name = 'sentence_with_number_entities_emotion',
                     llm = llm_async)

# This is an example of an external user-defined function (see Tutorial 0)
def binary_to_decimal(binary_number: str) -> int:
    '''Converts binary_number to integer of base 10'''
    return int(str(binary_number), 2)

# Initialise your Agent and assign functions
my_agent = AsyncAgent('Helpful assistant', 'You are a generalist agent', 
                      llm = llm_async).assign_functions([sentence_style, binary_to_decimal])

# Run the Agent
output = await my_agent.run('First convert binary string 1001 to a number, then generate me a happy sentence with that number and a ball')
```

# 4. Shared Variables

*"Because text is not enough"* - Anonymous

- `shared_variables` is a dictionary, that is initialised in Agent (default empty dictionary), and can be referenced by any function of the agent (including Inner Agents and their functions)
- This can be useful for non-text modalities (e.g. audio, pdfs, image) and lengthy text modalities, which we do not want to output into `subtasks_completed` directly
- To use, simply define an External Function with `shared_variables` as the first input variable, from which you can access and modify `shared_variables` directly
- The agent will also be able to be self-referenced in the External Function via `shared_variables['agent']`, so you can change the agent's internal parameters via `shared_variables`
- If the function has no output because the output is stored in `shared_variables`, the default return value will be `{'Status': 'Completed'}`

### Example External Function using `shared_variables`
```python
# Use shared_variables as input to your external function to access and modify the shared variables
def generate_quotes(shared_variables, number_of_quotes: int, category: str):
    ''' Generates number_of_quotes quotes about category '''
    # Retrieve from shared variables
    my_quote_list = shared_variables['Quote List']
    
    # Generate the quotes
    res = strict_json(system_prompt = f'''Generate {number_of_quotes} sentences about {category}. 
Do them in the format "<Quote> - <Person>", e.g. "The way to get started is to quit talking and begin doing. - Walt Disney"
Ensure your quotes contain only ' within the quote, and are enclosed by " ''',
                      user_prompt = '',
                      output_format = {'Quote List': f'list of {number_of_quotes} quotes, type: List[str]'},
                      llm = llm)
    
    my_quote_list.extend([f'Category: {category}. '+ x for x in res['Quote List']])
    
    # Store back to shared variables
    shared_variables['Quote List'] = my_quote_list
```

# 5. Global Context

- `Global Context` is a very powerful feature in AgentJo, as it allows the Agent to be updated with the latest environmental state before every decision it makes
- It also allows for learnings in `shared_variables` to be carried across tasks, making the Agent teachable and learn through experiences
- A recommended practice is to always store the learnings of the Agent during the External Function call, and reset the Agent after each task, so that `subtasks_completed` will be as short as possible to avoid confusion to the Agent

- There are two ways to use `Global Context`, and both can be used concurrently:
    - 1. `global_context`
        - If all you need in the global context is `shared_variables` without any modification to it, then you can use `global_context`
        - `global_context` is a string with `<shared_variables_name>` enclosed with `<>`. These <> will be replaced with the actual variable in `shared_variables`
    - 2. `get_global_context` 
        - `get_global_context` is a function that takes in the agent's internal parameters (self) and outputs a string to the LLM to append to the prompts of any LLM-based calls internally, e.g. `get_next_subtask`, `use_llm`, `reply_to_user`
    - You have full flexibility to access anything the agent knows and process the `shared_variables` as required and configure a global prompt to the agent
    
## Example for `global_context` : Inventory Manager
- We can use `Global Context` to keep track of inventory state
- We simply get the functions `add_item_to_inventory` and `remove_item_from_inventory` to modify the `shared_variable` named `Inventory`
- Note we can also put rule-based checks like checking if item is in inventory before removing inside the function
- Even after task reset, the Agent still knows the inventory because of `Global Context`

```python
def add_item_to_inventory(shared_variables, item: str) -> str:
    ''' Adds item to inventory, and returns outcome of action '''
    shared_variables['Inventory'].append(item)
    return f'{item} successfully added to Inventory'
    
def remove_item_from_inventory(shared_variables, item: str) -> str:
    ''' Removes item from inventory and returns outcome of action '''
    if item in shared_variables['Inventory']:
        shared_variables['Inventory'].remove(item)
        return f'{item} successfully removed from Inventory'
    else:
        return f'{item} not found in Inventory, unable to remove'
    
agent = Agent('Inventory Manager', 
              'Adds and removes items in Inventory. Only able to remove items if present in Inventory',
              shared_variables = {'Inventory': []},
              global_context = 'Inventory: <Inventory>', # Add in Global Context here with shared_variables Inventory
              llm = llm).assign_functions([add_item_to_inventory, remove_item_from_inventory])
```
    
# Other Features
- There are other features like Memory (Tutorial 3), Hierarchical Agents (Tutorial 4), CodeGen and External Function Interfacing (Tutorial 5), Conversation Class (Tutorial 6)
- These extend the baseline features of AgentJo and you are encouraged to take a look at the Tutorials for more information.

# Known Limitations
- There is an ongoing beta testing of YAML parsing (more efficient and reliable) using the `parse_yaml` function in StrictJSON repo. It is currently the default in AgentJo and is meant to be the more robust choice for longer text, nested outputs and code blocks.
- However, smaller LLMs are known to output JSON better than YAML.
- If `parse_yaml` does not work for you, to revert back to `strict_json`, simply set `llm_parser` to be `strict_json` and `llm_parser_async` to be `strict_json_async` when you declare your Agents and Functions.

# Contributing to the project

## Submitting a pull request
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Push your changes to your fork
5. Submit a pull request

# What are we looking out for?
1. Contributing Agent Wrappers and Memory Classes in contrib folder
2. Contribute Jupyter Notebooks in contrib folder showcasing what could be done with the framework for something useful. Let your imagination guide you, we look forward to see what you create
3. Other Known Limitations - Do test the framework out extensively and note its failure cases. We will see if we can address them, if not we will put them in Known Limitations.
4. (For the prompt engineer). If you could find a better way to make the prompts work, let us know directly - we do need to test this out across all Tutorial Jupyter Notebooks to make sure that it really works with existing datasets. Also, if you are using other LLMs beside OpenAI, and find the prompts do not work as well - try to rejig your own prompts and let us know as well!