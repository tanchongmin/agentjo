import copy
import importlib
import inspect
import os
import dill as pickle
import re
import subprocess
import sys

from termcolor import colored
import requests
from agentjo.base import strict_json
from agentjo.function import Function
from agentjo.base_async import strict_json_async
from agentjo.function import AsyncFunction
from agentjo.memory import AsyncMemory, Memory
from agentjo.utils import ensure_awaitable


class BaseAgent:
    def __init__(self, agent_name: str = 'Helpful Assistant',
                 agent_description: str = 'A generalist agent meant to help solve problems',
                 max_subtasks: int = 5,
                 summarise_subtasks_count: int = 5,
                 memory_bank = None,
                 shared_variables = None,
                 get_global_context = None,
                 global_context = '',
                 default_to_llm = True,
                 code_action = False,
                 verbose: bool = True,
                 debug: bool = False,
                 llm = None,
                 **kwargs): 
        ''' 
        Creates an LLM-based agent using description and outputs JSON based on output_format. 
        Agent does not answer in normal free-flow conversation, but outputs only concise task-based answers
        Design Philosophy:
        - Give only enough context needed to solve problem
        - Modularise components for greater accuracy of each component
        
        Inputs:
        - agent_name: String. Name of agent, hinting at what the agent does
        - agent_description: String. Short description of what the agent does
        - max_subtasks: Int. Default: 5. The maximum number of subtasks the agent can have
        - summarise_subtasks_count: Int. Default: 3. The maximum number of subtasks in Subtasks Completed before summary happens
        - memory_bank: class Dict[Memory]. Stores multiple types of memory for use by the agent. Customise the Memory config within the Memory class.
            - Key: `Function` (Already Implemented Natively) - Does RAG over Task -> Function mapping
            - Can add in more keys that would fit your use case. Retrieves similar items to task / overall plan (if able) for additional context in `get_next_subtasks()` and `use_llm()` function
            - For RAG over Documents, it is best done in a function of the Agent to retrieve more information when needed (so that we do not overload the Agent with information)
        - shared_variables. Dict. Default: None. Stores the variables to be shared amongst inner functions and agents. 
            If not empty, will pass this dictionary by reference down to the inner agents and functions
        - get_global_context. Function. Takes in self (agent variable) and returns the additional prompt (str) to be appended to `get_next_subtask` and `use_llm`. Allows for persistent agent states to be added to the prompt
        - global_context. String. Additional global context in string form. Put in variables to substitute for shared_variables using <>
        - default_to_llm. Bool. Default: True. Whether to default to use_llm function if there is no match to other functions. If False, use_llm will not be given to Agent
        - code_action. Bool. Default: False. Whether to use code as the only action space
        - verbose: Bool. Default: True. Whether to print out intermediate thought processes of the Agent
        - debug: Bool. Default: False. Whether to debug StrictJSON messages
        - llm: Function. The llm parameter that gets passed into Function/strict_json
        
        Inputs (optional):
        - **kwargs: Dict. Additional arguments you would like to pass on to the strict_json function
        
        '''
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.max_subtasks = max_subtasks
        self.summarise_subtasks_count = summarise_subtasks_count
        self.verbose = verbose
        self.default_to_llm = default_to_llm
        self.code_action = code_action
        self.get_global_context = get_global_context
        self.global_context = global_context

        self.debug = debug
        self.llm = llm
        
        # set shared variables
        if shared_variables is None:
            self.shared_variables = {}
        else:
            self.shared_variables = shared_variables
        self.init_shared_variables = copy.deepcopy(self.shared_variables)
        # append agent to shared variables, so that functions have access to it
        self.shared_variables['agent'] = self
        self.memory_bank = memory_bank
        
        # initialise the thoughts - this records the ReAct framework of Observation, Thoughts, Action at each step
        self.thoughts = []

        # reset agent's state
        self.reset()

        self.kwargs = kwargs

        # start with default of only llm as the function
        self.function_map = {}
        # stores all existing function descriptions - prevent duplicate assignment of functions
        self.fn_description_list = []
        
    def reset(self):
        ''' Resets agent state, including resetting subtasks_completed '''
        self.assign_task('No task assigned')
        self.subtasks_completed = {}
        # reset all thoughts
        self.thoughts = []
    
            
    def assign_task(self, task: str, overall_task: str = ''):
        ''' Assigns a new task to this agent. Also treats this as the meta agent now that we have a task assigned '''
        self.task = task
        self.overall_task = task
        # if there is a meta agent's task, add this to overall task
        if overall_task != '':
            self.overall_task = overall_task
            
        self.task_completed = False
        self.overall_plan = None
        
    def save_agent(self, filename: str):
        ''' Saves the entire agent to filename for reuse next time '''
        
        if not isinstance(filename, str):
            if filename[-4:] != '.pkl':
                raise Exception('Filename is not ending with .pkl')
            return
            
        # Open a file in write-binary mode
        with open(filename, 'wb') as file:
            # Use pickle.dump() to save the dictionary to the file
            pickle.dump(self, file)

        print(f"Agent saved to {filename}")
    
    def load_agent(self, filename: str):
        ''' Loads the entire agent from filename '''
        
        if not isinstance(filename, str):
            if filename[-4:] != '.pkl':
                raise Exception('Filename is not ending with .pkl')
            return
        
        with open(filename, 'rb') as file:
            self = pickle.load(file)
            print(f"Agent loaded from {filename}")
            return self
        
    def status(self):
        ''' Prints prettily the update of the agent's status. 
        If you would want to reference any agent-specific variable, just do so directly without calling this function '''
        print('Agent Name:', self.agent_name)
        print('Agent Description:', self.agent_description)
        print('Available Functions:', list(self.function_map.keys()))
        if len(self.shared_variables) > 0:
            print('Shared Variables:', list(self.shared_variables.keys()))
        print(colored(f'Task: {self.task}', 'green', attrs = ['bold']))
        if len(self.subtasks_completed) == 0: 
            print(colored("Subtasks Completed: None", 'blue', attrs = ['bold']))
        else:
            print(colored('Subtasks Completed:', 'black', attrs = ['bold']))
            for key, value in self.subtasks_completed.items():
                print(colored(f"Subtask: {key}", 'blue', attrs = ['bold']))
                print(f'{value}\n')
        print('Is Task Completed:', self.task_completed)
        
    def remove_function(self, function_name: str):
        ''' Removes a function from the agent '''
        if function_name in self.function_map:
            function = self.function_map[function_name]
            # remove actual function from memory bank
            if function_name not in ['use_llm', 'end_task']:
                self.memory_bank['Function'].remove(function)
            # remove function description from fn_description_list
            self.fn_description_list.remove(function.fn_description)
            # remove function from function map
            del self.function_map[function_name]
    
    def list_functions(self, fn_list = None) -> list:
        ''' Returns the list of functions available to the agent. If fn_list is given, restrict the functions to only those in the list '''
        if fn_list is not None and len(fn_list) < len(self.function_map):
            if self.verbose:
                print('Filtered Function Names:', ', '.join([name for name, function in self.function_map.items() if function in fn_list]))
            return [f'Name: {name}\n' + str(function) for name, function in self.function_map.items() if function in fn_list]
        else:
            return [f'Name: {name}\n' + str(function) for name, function in self.function_map.items()]                       
    
    def print_functions(self):
        ''' Prints out the list of functions available to the agent '''
        functions = self.list_functions()
        print('\n'.join(functions))    
    
    def add_subtask_result(self, subtask, result):
        ''' Adds the subtask and result to subtasks_completed
        Keep adding (num) to subtask str if there is duplicate '''
        subtask_str = str(subtask)
        count = 2
        
        # keep adding count until we have a unique id
        while subtask_str in self.subtasks_completed:
            subtask_str = str(subtask) + f'({count})'
            count += 1
            
        self.subtasks_completed[subtask_str] = result      
       
            
    def remove_last_subtask(self):
        ''' Removes last subtask in subtask completed. Useful if you want to retrace a step '''
        if len(self.subtasks_completed) > 0:
            removed_item = self.subtasks_completed.popitem()
        if self.verbose:
            print(f'Removed last subtask from subtasks_completed: {removed_item}')  
                
    # Alternate names
    list_function = list_functions
    list_tools = list_functions
    list_tool = list_functions
    print_function = print_functions
    print_tools = print_functions
    print_tool = print_functions
    remove_tool = remove_function
  
###########################
## Sync Version of Agent ##
###########################
class Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_memory = Memory(top_k = 5, mapper = lambda x: x.fn_name + ': ' + x.fn_description, approach = 'retrieve_by_ranker', llm = self.llm)
        if self.memory_bank is None:
            self.memory_bank = {'Function': self.default_memory}
            self.memory_bank['Function'].reset()
            
            # adds the use llm function
        if self.default_to_llm:
            self.assign_functions([Function(fn_name = 'use_llm', 
                                        fn_description = f'For general tasks. Used only when no other function can do the task', 
                                        is_compulsory = True,
                                        output_format = {"Output": "Output of LLM"})])
        # adds the end task function
        self.assign_functions([Function(fn_name = 'end_task',
                                       fn_description = 'Passes the final output to the user',
                                       is_compulsory = True,
                                       output_format = {})])
        
        
    def query(self, query: str, output_format: dict, provide_function_list: bool = False, task: str = ''):
        ''' Queries the agent with a query and outputs in output_format. 
        If task is provided, we will filter the functions according to the task
        If you want to provide the agent with the context of functions available to it, set provide_function_list to True (default: False)
        If task is given, then we will use it to do RAG over functions'''
        
        # if we have a task to focus on, we can filter the functions (other than use_llm and end_task) by that task
        filtered_fn_list = None
        if task != '':
            # filter the functions
            filtered_fn_list = self.memory_bank['Function'].retrieve(task)
            
            # add back compulsory functions (default: use_llm, end_task) if present in function_map
            for name, function in self.function_map.items():
                if function.is_compulsory:
                    filtered_fn_list.append(function)
                
        # add in global context string and replace it with shared_variables as necessary
        global_context_string = self.global_context
        matches = re.findall(r'<(.*?)>', global_context_string)
        for match in matches:
            if match in self.shared_variables:
                global_context_string = global_context_string.replace(f'<{match}>', str(self.shared_variables[match]))
                
        # add in the global context function's output
        global_context_output = self.get_global_context(self) if self.get_global_context is not None else ''
            
        global_context = ''
        # Add in global context if present
        if global_context_string != '' or global_context_output != '':
            global_context = 'Global Context:\n```\n' + global_context_string + '\n' + global_context_output + '```\n'
        
        user_prompt = f'''You are an agent named {self.agent_name} with the following description: ```{self.agent_description}```\n'''
        if provide_function_list:
            user_prompt += f"You have the following Equipped Functions available:\n```{self.list_functions(filtered_fn_list)}```\n"
        user_prompt += global_context
        user_prompt += query
        
        res = strict_json(system_prompt = '',
        user_prompt = user_prompt,
        output_format = output_format, 
        verbose = self.debug,
        llm = self.llm,
        **self.kwargs)

        return res
      
    ## Functions for function calling ##
    def assign_functions(self, function_list: list):
        ''' Assigns a list of functions to be used in function_map '''
        if not isinstance(function_list, list):
            function_list = [function_list]
            
        for function in function_list:
            # If this function is an Agent, parse it accordingly
            if isinstance(function, BaseAgent):
                function = function.to_function(self)
            
            # do automatic conversion of function to Function class (this is in base.py)
            if not isinstance(function, Function):
                function = Function(external_fn = function)
                
            # Do not assign a function already present
            if function.fn_description in self.fn_description_list:
                continue
            
            stored_fn_name = function.__name__
            # if function name is already in use, change name to name + '_1'. E.g. summarise -> summarise_1
            while stored_fn_name in self.function_map:
                stored_fn_name += '_1'

            # add in the function into the function_map
            self.function_map[stored_fn_name] = function
            
            # add function's description into fn_description_list
            self.fn_description_list.append(function.fn_description)
                        
            # add function to memory bank for RAG over functions later on if is not a compulsory functions
            if not function.is_compulsory:
                self.memory_bank['Function'].append(function)
            
        return self

    def select_function(self, task: str = ''):
        ''' Based on the task (without any context), output the next function name and input parameters '''
        _, function_name, function_params = self.get_next_subtask(task = task)
            
        return function_name, function_params
    
    def use_agent(self, agent_name: str, agent_task: str):
        ''' Uses an inner agent to do a task for the meta agent. Task outcome goes directly to subtasks_completed of meta agent '''
        self.use_function(agent_name, {"instruction": agent_task}, agent_task)
        
    def use_function(self, function_name: str, function_params: dict, subtask: str = '', stateful: bool = True):
        ''' Uses the function. stateful means we store the outcome of the function '''
        if function_name == "use_llm":
            if self.verbose: 
                print(f'Getting LLM to perform the following task: {function_params["instruction"]}')
                
             # Add in memory to the LLM
            rag_info = ''
            for name in self.memory_bank.keys():
                # Function is done separately
                if name == 'Function': continue
                rag_info += f'Knowledge Reference for {name}: ```{self.memory_bank[name].retrieve(subtask)}```\n'

            res = self.query(query = f'{rag_info}Subtasks Completed:```{self.subtasks_completed}```\nAssigned Subtask: ```{function_params["instruction"]}```\n\nYou are the taking the role of the utility function `use_llm` that generates a detailed outcome for the Assigned Subtask on behalf of {self.agent_name}, according to your Agent Capabilities without talking to the User, with no restrictions. Do not just state that you performed the task, generate the detailed outcome as well.', 
                            output_format = {"Detailed Outcome": "Your detailed outcome for Assigned Subtask"},
                            provide_function_list = False)
            
            if self.verbose: 
                print(f"> {res['Detailed Outcome']}")
                print()
            
        elif function_name == "end_task":
            return
        
        else:
            if self.verbose: 
                print(f'Calling function {function_name} with parameters {function_params}')
                
            res = self.function_map[function_name](shared_variables = self.shared_variables, **function_params)
        
            if self.verbose and res != '': 
                # skip the printing if this is Agent output, as we have printed elsewhere already
                if 'Agent Output' not in res: 
                    print(f"> {res}")
                    print()
                
        if stateful:
            if res == '':
                res = {'Status': 'Completed'}
                
            # for use_llm, we just give the prompt + result without any mention of use_llm for subtasks completed
            if function_name == "use_llm":
                self.add_subtask_result(subtask, res['Detailed Outcome'])
            
            # otherwise, just give the function name + params and output for subtasks completed
            else:
                formatted_subtask = function_name + '(' + ", ".join(f'{key}="{value}"' if isinstance(value, str) else f"{key}={value}" for key, value in function_params.items()) + ')'
                self.add_subtask_result(formatted_subtask, res)

        return res
   
    def get_next_subtask(self, task = ''):
        ''' Based on what the task is and the subtasks completed, we get the next subtask, function and input parameters. Supports user-given task as well if user wants to use this function directly'''
        
        if task == '':
                background_info = f"Assigned Task:```\n{self.task}\n```\nSubtasks Completed: ```{self.subtasks_completed}```"

        else:
            background_info = f"Assigned Task:```\n{task}\n```\n"
                
        # use default agent plan if task is not given
        task = self.task if task == '' else task
            
        # Add in memory to the Agent
        rag_info = ''
        for name in self.memory_bank.keys():
            # Function RAG is done separately in self.query()
            if name == 'Function': continue
            rag_info += f'Knowledge Reference for {name}: ```{self.memory_bank[name].retrieve(task)}```\n'
                
        # First select the Equipped Function
        res = self.query(query = f'''{background_info}{rag_info}\nBased on everything before, provide suitable Observation and Thoughts, and also generate the Current Subtask and the corresponding Equipped Function Name to complete a part of Assigned Task.
You are only given the Assigned Task from User with no further inputs. Only focus on the Assigned Task and do not do more than required. 
End Task if Assigned Task is completed.''',
         output_format = {"Observation": "Reflect on what has been done in Subtasks Completed for Assigned Task", 
                          "Thoughts": "Brainstorm how to complete remainder of Assigned Task only given Observation", 
                          "Current Subtask": "What to do now in detail with all context provided that can be done by one Equipped Function for Assigned Task", 
                          "Equipped Function Name": "Name of Equipped Function to use for Current Subtask"},
             provide_function_list = True,
             task = task)

        if self.verbose:
            print(colored(f"Observation: {res['Observation']}", 'black', attrs = ['bold']))
            print(colored(f"Thoughts: {res['Thoughts']}", 'green', attrs = ['bold']))
            
        # end task if equipped function is incorrect
        if res["Equipped Function Name"] not in self.function_map:
            res["Equipped Function Name"] = "end_task"
                
        # If equipped function is use_llm, or end_task, we don't need to do another query
        cur_function = self.function_map[res["Equipped Function Name"]]
        
        # Do an additional check to see if we are using code action space
        if self.code_action and res['Equipped Function Name'] != 'end_task' and 'python_generate_and_run_code_tool' in self.function_map:
            res["Equipped Function Name"] = 'python_generate_and_run_code_tool'
            res['Equipped Function Inputs'] = {'instruction': res['Current Subtask']}
        elif res["Equipped Function Name"] == 'use_llm':
            res['Equipped Function Inputs'] = {'instruction': res['Current Subtask']}
        elif res['Equipped Function Name'] == 'end_task':
            res['Equipped Function Inputs'] = {}   
        # Otherwise, if it is only the instruction, no type check needed, so just take the instruction
        elif len(cur_function.variable_names) == 1 and cur_function.variable_names[0].lower() == "instruction":
            res['Equipped Function Inputs'] = {'instruction': res['Current Subtask']}
            
        # Otherwise, do another query to get type-checked input parameters and ensure all input fields are present
        else:
            input_format = {}
            fn_description = cur_function.fn_description
            matches = re.findall(r'<(.*?)>', fn_description)
            
            # do up an output format dictionary to use to get LLM to output exactly based on keys and types needed
            for match in matches:
                if ':' in match:
                    first_part, second_part = match.split(':', 1)
                    input_format[first_part] = f'A suitable value, type: {second_part}'
                else:
                    input_format[match] = 'A suitable value'
                    
            # if there is no input, then do not need LLM to extract out function's input
            if input_format == {}:
                res["Equipped Function Inputs"] = {}
                    
            else:    
                res2 = self.query(query = f'''{background_info}{rag_info}\n\nCurrent Subtask: ```{res["Current Subtask"]}```\nEquipped Function Details: ```{str(cur_function)}```\nOutput suitable values for Inputs to Equipped Function to fulfil Current Subtask\nInput fields are: {list(input_format.keys())}''',
                             output_format = input_format,
                             provide_function_list = False)
                
                # store the rest of the function parameters
                res["Equipped Function Inputs"] = res2
                
        # Add in output to the thoughts
        self.thoughts.append(res)
            
        return res["Current Subtask"], res["Equipped Function Name"], res["Equipped Function Inputs"]
  
        
    def summarise_subtasks_completed(self, task: str = ''):
        ''' Summarise the subtasks_completed list according to task '''

        output = self.reply_user(task)
        # Create a new summarised subtasks completed list
        self.subtasks_completed = {f"Current Results for '{task}'": output}
        
    def reply_user(self, query: str = '', stateful: bool = True, verbose: bool = True):
        ''' Generate a reply to the user based on the query / agent task and subtasks completed
        If stateful, also store this interaction into the subtasks_completed
        If verbose is given, can also override the verbosity of this function'''
        
        my_query = self.task if query == '' else query
            
        res = self.query(query = f'Subtasks Completed: ```{self.subtasks_completed}```\nAssigned Task: ```{my_query}```\nRespond to the Assigned Task using information from Global Context and Subtasks Completed only. Be factual and do not generate any new information. Be detailed and give all information available relevant for the Assigned Task in your Assigned Task Response', 
                                    output_format = {"Assigned Task Response": "Detailed Response"},
                                    provide_function_list = False)
        
        res = res["Assigned Task Response"]
        
        if self.verbose and verbose:
            print(res)
        
        if stateful:
            self.add_subtask_result(my_query, res)
        
        return res

    ## this gets the agent to answer based on a certain output_format
    def answer(self, query, output_format = {'Answer': 'Concise Answer'}):
        ''' This answers the user based on output_format
        query (str): The question the user wants to ask
        output_format (dict): The output format in a dictionary'''

        return self.query(
f'''Answer the following query according to the subtasks completed: ```{query}```\n
Subtasks Completed: ```{self.subtasks_completed}```
Be concise and just give the answer with no explanation required.''', 
                               output_format = output_format)

    def run(self, task: str = '', overall_task: str = '', num_subtasks: int = 0) -> list:
        ''' Attempts to do the task using LLM and available functions
        Loops through and performs either a function call or LLM call up to num_subtasks number of times
        If overall_task is filled, then we store it to pass to the inner agents for more context '''
            
        # Assign the task
        if task != '':
            self.task_completed = False
            # If meta agent's task is here as well, assign it too
            if overall_task != '':
                self.assign_task(task, overall_task)
            else:
                self.assign_task(task)
            
        # check if we need to override num_steps
        if num_subtasks == 0:
            num_subtasks = self.max_subtasks
        
        # if task completed, then exit
        if self.task_completed: 
            if self.verbose:
                print('Task already completed!\n')
                print('Subtasks completed:')
                for key, value in self.subtasks_completed.items():
                    print(f"Subtask: {key}\n{value}\n")
                    
        else:
            # otherwise do the task
            for i in range(num_subtasks):           
                # Determine next subtask, or if task is complete. Always execute if it is the first subtask
                subtask, function_name, function_params = self.get_next_subtask()
                if function_name == 'end_task':
                    self.task_completed = True
                    if self.verbose:
                        print(colored(f"Subtask identified: End Task", "blue", attrs=['bold']))
                        print('Task completed successfully!\n')
                    break
                    
                if self.verbose: 
                    print(colored(f"Subtask identified: {subtask}", "blue", attrs=['bold']))

                # Execute the function for next step
                res = self.use_function(function_name, function_params, subtask)
                
                # Summarise Subtasks Completed if necessary
                if len(self.subtasks_completed) > self.summarise_subtasks_count:
                    print('### Auto-summarising Subtasks Completed (Change frequency via `summarise_subtasks_count` variable) ###')
                    self.summarise_subtasks_completed(f'progress for {self.overall_task}')
                    print('### End of Auto-summary ###\n')
          
        return list(self.subtasks_completed.values())
    
    ## This is for Multi-Agent uses
    def to_function(self, meta_agent):
        ''' Converts the agent to a function so that it can be called by another agent
        The agent will take in an instruction, and output the result after processing'''

        # makes the agent appear as a function that takes in an instruction and outputs the executed instruction
        my_fn = Function(fn_name = self.agent_name,
                             fn_description = f'Agent Description: ```{self.agent_description}```\nExecutes the given <instruction>',
                             output_format = {"Agent Output": "Output of instruction"},
                             external_fn = Agent_External_Function(self, meta_agent))
        
        return my_fn
    
    def assign_agents(self, agent_list: list):
        ''' Assigns a list of Agents to the main agent, passing in the meta agent as well '''
        if not isinstance(agent_list, list):
            agent_list = [agent_list]
        self.assign_functions([agent.to_function(self) for agent in agent_list])
        return self
    
    ## Credit: sebbecht
    def wrap_function(self, func, before_hook: list = [], after_hook: list = []):
        """
        Wraps a base agent function with hooks to be forcefully executed before and after the base function. 
        Hooks are executed in the order they are passed in.
        Hooks do not become part of the agent's assigned functions.
        
        Args:
            func: The base agent function to wrap. Can be any agent's internal function like run, reply_user
            before_hook: List of callable functions to execute before the base function
            after_hook: List of callable functions to execute after the base function
        """
        # Make sure use_function is not wrapped as it will cause infinite loops
        if func == 'use_function':
            raise Exception("use_function cannot be wrapped")
        
        # if functions are not already Function objects, convert all hooks to Function objects. If they are BaseAgent, convert them to Function objects
        before_hook = [hook.to_function(self) if isinstance(hook, BaseAgent) else Function(external_fn=hook) for hook in before_hook]
        after_hook = [hook.to_function(self) if isinstance(hook, BaseAgent) else Function(external_fn=hook) for hook in after_hook]
                
        def infer_function_parameters(function: Function):
            input_format = {}
            fn_description = function.fn_description
            matches = re.findall(r'<(.*?)>', fn_description)
            
            # do up an output format dictionary to use to get LLM to output exactly based on keys and types needed
            for match in matches:
                if ':' in match:
                    first_part, second_part = match.split(':', 1)
                    input_format[first_part] = f'A suitable value, type: {second_part}'
                else:
                    input_format[match] = 'A suitable value'
                    
            # if there is no input, then do not need LLM to extract out function's input
            if input_format == {}:
                function_params = {}
                    
            else:
                background_info = f"Assigned Task:```\n{self.task}\n```\nSubtasks Completed: ```{self.subtasks_completed}```"
                # Add in memory to the Agent
                rag_info = ''
                for name in self.memory_bank.keys():
                    # Function RAG is done separately in self.query()
                    if name == 'Function': continue
                rag_info += f'Knowledge Reference for {name}: ```{self.memory_bank[name].retrieve(self.task)}```\n'    

                function_params = self.query(query = f'''{background_info}{rag_info}\n\n```\nEquipped Function Details: ```{str(function)}```\nOutput suitable values for Inputs to Equipped Function to fulfil Current Subtask\nInput fields are: {list(input_format.keys())}''',
                            output_format = input_format,
                            provide_function_list = False)
            return function_params
        
        try:    
            # Get the original base function
            original_func = getattr(self, func)
        except:
            raise Exception(f"Base function: {func} not found in agent {self.agent_name}")
        
        def wrapped_function(*args, **kwargs):
            # Special handling for run function to ensure task is assigned before hooks
            if func == 'run' and len(args) > 0:
                self.assign_task(args[0], args[1] if len(args) > 1 else '')
            
            # Execute before hooks
            for hook in before_hook:
                self.assign_functions([hook])
                input_params = infer_function_parameters(hook)
                self.use_function(hook.fn_name, input_params)
                self.remove_function(hook.fn_name)
                
            # Execute original function
            result = original_func(*args, **kwargs)
            
            # Execute after hooks
            for hook in after_hook:
                self.assign_functions([hook])
                input_params = infer_function_parameters(hook)
                self.use_function(hook.fn_name, input_params)
                self.remove_function(hook.fn_name)
                
            return result
        
        # Replace the original function with wrapped version
        setattr(self, func, wrapped_function)
    
    ## Function aliaises
    assign_function = assign_functions
    assign_tool = assign_functions
    assign_tools = assign_functions
    select_tool = select_function
    use_tool = use_function
    assign_agent = assign_agents
    
############################
## Async Version of Agent ##
############################

class AsyncAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_memory = AsyncMemory(
            top_k=5, 
            mapper=lambda x: (x.fn_name or '') + ': ' + (x.fn_description or ''), 
            approach='retrieve_by_ranker',
            llm = self.llm
        )
        if self.memory_bank is None:
            self.memory_bank = {'Function': self.default_memory}
            self.memory_bank['Function'].reset()
        if not isinstance(self.memory_bank['Function'], AsyncMemory):
            raise Exception('Sync memory not allowed for Async Agent')
        if self.default_to_llm:
            self.assign_functions([AsyncFunction(fn_name = 'use_llm', 
                                        fn_description = f'For general tasks. Used only when no other function can do the task', 
                                        is_compulsory = True,
                                        output_format = {"Output": "Output of LLM"})])
        # adds the end task function
        self.assign_functions([AsyncFunction(fn_name = 'end_task',
                                       fn_description = 'Passes the final output to the user',
                                       is_compulsory = True,
                                       output_format = {})])
        
    async def query(self, query: str, output_format: dict, provide_function_list: bool = False, task: str = ''):
        ''' Queries the agent with a query and outputs in output_format. 
        If task is provided, we will filter the functions according to the task
        If you want to provide the agent with the context of functions available to it, set provide_function_list to True (default: False)
        If task is given, then we will use it to do RAG over functions'''
        
        # if we have a task to focus on, we can filter the functions (other than use_llm and end_task) by that task
        filtered_fn_list = None
        if task != '':
            # filter the functions
            filtered_fn_list = await self.memory_bank['Function'].retrieve(task)
            
            # add back compulsory functions (default: use_llm, end_task) if present in function_map
            for name, function in self.function_map.items():
                if function.is_compulsory:
                    filtered_fn_list.append(function)
                
        # add in global context string and replace it with shared_variables as necessary
        global_context_string = self.global_context
        matches = re.findall(r'<(.*?)>', global_context_string)
        for match in matches:
            if match in self.shared_variables:
                global_context_string = global_context_string.replace(f'<{match}>', str(self.shared_variables[match]))
                
        # add in the global context function's output
        global_context_output = self.get_global_context(self) if self.get_global_context is not None else ''
            
        global_context = ''
        # Add in global context if present
        if global_context_string != '' or global_context_output != '':
            global_context = 'Global Context:\n```\n' + global_context_string + '\n' + global_context_output + '```\n'
        
        user_prompt = f'''You are an agent named {self.agent_name} with the following description: ```{self.agent_description}```\n'''
        if provide_function_list:
            user_prompt += f"You have the following Equipped Functions available:\n```{self.list_functions(filtered_fn_list)}```\n"
        user_prompt += global_context
        user_prompt += query
        
        res = await strict_json_async(system_prompt = '',
        user_prompt = user_prompt,
        output_format = output_format, 
        verbose = self.debug,
        llm = self.llm,
        **self.kwargs)

        return res
       
    ## Functions for function calling ##
    def assign_functions(self, function_list: list):
        ''' Assigns a list of functions to be used in function_map '''
        if not isinstance(function_list, list):
            function_list = [function_list]
            
        for function in function_list:
            # If this function is an Agent, parse it accordingly
            if isinstance(function, BaseAgent):
                function = function.to_function(self)
                
            # do automatic conversion of function to Function class (this is in base.py)
            if not isinstance(function, AsyncFunction):
                function = AsyncFunction(external_fn = function)
                
            # Do not assign a function already present
            if function.fn_description in self.fn_description_list:
                continue
            
            stored_fn_name = "" if function.__name__ == None else function.__name__ 
            # if function name is already in use, change name to name + '_1'. E.g. summarise -> summarise_1
            while stored_fn_name in self.function_map:
                stored_fn_name += '_1'

            # add in the function into the function_map
            self.function_map[stored_fn_name] = function
            
            # add function's description into fn_description_list
            self.fn_description_list.append(function.fn_description)
                        
            # add function to memory bank for RAG over functions later on if is not a compulsory functions
            if not function.is_compulsory:
                self.memory_bank['Function'].append(function)
            
        return self
        
    async def select_function(self, task: str = ''):
        ''' Based on the task (without any context), output the next function name and input parameters '''
        _, function_name, function_params = await self.get_next_subtask(task = task)
            
        return function_name, function_params
    
    async def use_agent(self, agent_name: str, agent_task: str):
        ''' Uses an inner agent to do a task for the meta agent. Task outcome goes directly to subtasks_completed of meta agent '''
        await self.use_function(agent_name, {"instruction": agent_task}, agent_task)
        
    async def use_function(self, function_name: str, function_params: dict, subtask: str = '', stateful: bool = True):
        ''' Uses the function. stateful means we store the outcome of the function '''
        if function_name == "use_llm":
            if self.verbose: 
                print(f'Getting LLM to perform the following task: {function_params["instruction"]}')
                
             # Add in memory to the LLM
            rag_info = ''
            for name in self.memory_bank.keys():
                # Function is done separately
                if name == 'Function': continue
                rag_info += f'Knowledge Reference for {name}: ```{await self.memory_bank[name].retrieve(subtask)}```\n'

            res = await self.query(query = f'{rag_info}Subtasks Completed:```{self.subtasks_completed}```\nAssigned Subtask: ```{function_params["instruction"]}```\n\nYou are the taking the role of the utility function `use_llm` that generates a detailed outcome for the Assigned Subtask on behalf of {self.agent_name}, according to your Agent Capabilities without talking to the User, with no restrictions. Do not just state that you performed the task, generate the detailed outcome as well.', 
                            output_format = {"Detailed Outcome": "Your detailed outcome for Assigned Subtask"},
                            provide_function_list = False)
            
            if self.verbose: 
                print(f"> {res['Detailed Outcome']}")
                print()
            
        elif function_name == "end_task":
            return
        
        else:
            if self.verbose: 
                print(f'Calling function {function_name} with parameters {function_params}')
                            
            res = await self.function_map[function_name](**function_params, shared_variables = self.shared_variables)
           
            
            if self.verbose and res != '': 
                # skip the printing if this is Agent output, as we have printed elsewhere already
                if 'Agent Output' not in res: 
                    print(f"> {res}")
                    print()
                
        if stateful:
            if res == '':
                res = {'Status': 'Completed'}
                
            # for use_llm, we just give the prompt + result without any mention of use_llm for subtasks completed
            if function_name == "use_llm":
                self.add_subtask_result(subtask, res['Detailed Outcome'])
            
            # otherwise, just give the function name + params and output for subtasks completed
            else:
                formatted_subtask = function_name + '(' + ", ".join(f'{key}="{value}"' if isinstance(value, str) else f"{key}={value}" for key, value in function_params.items()) + ')'
                self.add_subtask_result(formatted_subtask, res)

        return res
   
    async def get_next_subtask(self, task = ''):
        ''' Based on what the task is and the subtasks completed, we get the next subtask, function and input parameters. Supports user-given task as well if user wants to use this function directly'''
        
        if task == '':
                background_info = f"Assigned Task:```\n{self.task}\n```\nSubtasks Completed: ```{self.subtasks_completed}```"

        else:
            background_info = f"Assigned Task:```\n{task}\n```\n"
                
        # use default agent plan if task is not given
        task = self.task if task == '' else task
            
        # Add in memory to the Agent
        rag_info = ''
        for name in self.memory_bank.keys():
            # Function RAG is done separately in self.query()
            if name == 'Function': continue
            rag_info += f'Knowledge Reference for {name}: ```{await self.memory_bank[name].retrieve(task)}```\n'
                
        # First select the Equipped Function
        res = await self.query(query = f'''{background_info}{rag_info}\nBased on everything before, provide suitable Observation and Thoughts, and also generate the Current Subtask and the corresponding Equipped Function Name to complete a part of Assigned Task.
You are only given the Assigned Task from User with no further inputs. Only focus on the Assigned Task and do not do more than required.
End Task if Assigned Task is completed.''',
         output_format = {"Observation": "Reflect on what has been done in Subtasks Completed for Assigned Task", 
                          "Thoughts": "Brainstorm how to complete remainder of Assigned Task only given Observation", 
                          "Current Subtask": "What to do now in detail with all context provided that can be done by one Equipped Function for Assigned Task", 
                          "Equipped Function Name": "Name of Equipped Function to use for Current Subtask"},
             provide_function_list = True,
             task = task)

        if self.verbose:
            print(colored(f"Observation: {res['Observation']}", 'black', attrs = ['bold']))
            print(colored(f"Thoughts: {res['Thoughts']}", 'green', attrs = ['bold']))
            
        # end task if equipped function is incorrect
        if res["Equipped Function Name"] not in self.function_map:
            res["Equipped Function Name"] = "end_task"
                
        # If equipped function is use_llm, or end_task, we don't need to do another query
        cur_function = self.function_map[res["Equipped Function Name"]]
        
        # Do an additional check to see if we should use code
        if self.code_action and res["Equipped Function Name"] != 'end_task' and 'python_generate_and_run_code_tool' in self.function_map:
            res["Equipped Function Name"] = 'python_generate_and_run_code_tool'
            res['Equipped Function Inputs'] = {'instruction': res['Current Subtask']}
        elif res["Equipped Function Name"] == 'use_llm':
            res['Equipped Function Inputs'] = {'instruction': res['Current Subtask']}
        elif res['Equipped Function Name'] == 'end_task':
            res['Equipped Function Inputs'] = {}
        # Otherwise, if it is only the instruction, no type check needed, so just take the instruction
        elif len(cur_function.variable_names) == 1 and cur_function.variable_names[0].lower() == "instruction":
            res['Equipped Function Inputs'] = {'instruction': res['Current Subtask']}
            
        # Otherwise, do another query to get type-checked input parameters and ensure all input fields are present
        else:
            input_format = {}
            fn_description = cur_function.fn_description
            matches = re.findall(r'<(.*?)>', fn_description)
            
            # do up an output format dictionary to use to get LLM to output exactly based on keys and types needed
            for match in matches:
                if ':' in match:
                    first_part, second_part = match.split(':', 1)
                    input_format[first_part] = f'A suitable value, type: {second_part}'
                else:
                    input_format[match] = 'A suitable value'
                    
            # if there is no input, then do not need LLM to extract out function's input
            if input_format == {}:
                res["Equipped Function Inputs"] = {}
                    
            else:    
                res2 = await self.query(query = f'''{background_info}{rag_info}\n\nCurrent Subtask: ```{res["Current Subtask"]}```\nEquipped Function Details: ```{str(cur_function)}```\nOutput suitable values for Inputs to Equipped Function to fulfil Current Subtask\nInput fields are: {list(input_format.keys())}''',
                             output_format = input_format,
                             provide_function_list = False)
                
                # store the rest of the function parameters
                res["Equipped Function Inputs"] = res2
                
        # Add in output to the thoughts
        self.thoughts.append(res)
            
        return res["Current Subtask"], res["Equipped Function Name"], res["Equipped Function Inputs"]
        
    async def summarise_subtasks_completed(self, task: str = ''):
        ''' Summarise the subtasks_completed list according to task '''

        output = await self.reply_user(task)
        # Create a new summarised subtasks completed list
        self.subtasks_completed = {f"Current Results for '{task}'": output}
        
    async def reply_user(self, query: str = '', stateful: bool = True, verbose: bool = True):
        ''' Generate a reply to the user based on the query / agent task and subtasks completed
        If stateful, also store this interaction into the subtasks_completed
        If verbose is given, can also override the verbosity of this function'''
        
        my_query = self.task if query == '' else query
            
        res = await self.query(query = f'Subtasks Completed: ```{self.subtasks_completed}```\nAssigned Task: ```{my_query}```\nRespond to the Assigned Task in detail using information from Global Context and Subtasks Completed only. Be factual and do not generate any new information. Be detailed and give all information available relevant for the Assigned Task in your Assigned Task Response', 
                                    output_format = {"Assigned Task Response": "Detailed Response"},
                                    provide_function_list = False)
        
        res = res["Assigned Task Response"]
        
        if self.verbose and verbose:
            print(res)
        
        if stateful:
            self.add_subtask_result(my_query, res)
        
        return res

    ## this gets the agent to answer based on a certain output_format
    async def answer(self, query, output_format = {'Answer': 'Concise Answer'}):
        ''' This answers the user based on output_format
        query (str): The query the user wants to ask
        output_format (dict): The output format in a dictionary'''
        
        return await self.query(
f'''Answer the following query according to the subtasks completed: ```{query}```\n
Subtasks Completed: ```{self.subtasks_completed}```
Be concise and just give the answer with no explanation required.''', 
                               output_format = output_format)

    async def run(self, task: str = '', overall_task: str = '', num_subtasks: int = 0) -> list:
        ''' Attempts to do the task using LLM and available functions
        Loops through and performs either a function call or LLM call up to num_subtasks number of times
        If overall_task is filled, then we store it to pass to the inner agents for more context '''
            
        # Assign the task
        if task != '':
            self.task_completed = False
            # If meta agent's task is here as well, assign it too
            if overall_task != '':
                self.assign_task(task, overall_task)
            else:
                self.assign_task(task)
            
        # check if we need to override num_steps
        if num_subtasks == 0:
            num_subtasks = self.max_subtasks
        
        # if task completed, then exit
        if self.task_completed: 
            if self.verbose:
                print('Task already completed!\n')
                print('Subtasks completed:')
                for key, value in self.subtasks_completed.items():
                    print(f"Subtask: {key}\n{value}\n")
                    
        else:
            # otherwise do the task
            for i in range(num_subtasks):           
                # Determine next subtask, or if task is complete. Always execute if it is the first subtask
                subtask, function_name, function_params = await self.get_next_subtask()
                if function_name == 'end_task':
                    self.task_completed = True
                    if self.verbose:
                        print(colored(f"Subtask identified: End Task", "blue", attrs=['bold']))
                        print('Task completed successfully!\n')
                    break
                    
                if self.verbose: 
                    print(colored(f"Subtask identified: {subtask}", "blue", attrs=['bold']))

                # Execute the function for next step
                res = await self.use_function(function_name, function_params, subtask)
                
                # Summarise Subtasks Completed if necessary
                if len(self.subtasks_completed) > self.summarise_subtasks_count:
                    print('### Auto-summarising Subtasks Completed (Change frequency via `summarise_subtasks_count` variable) ###')
                    await self.summarise_subtasks_completed(f'progress for {self.overall_task}')
                    print('### End of Auto-summary ###\n')
          
        return list(self.subtasks_completed.values())

    ## Credit: sebbecht
    async def wrap_function(self, func, before_hook: list = [], after_hook: list = []):
        """
        Wraps a base agent function with hooks to be forcefully executed before and after the base function. 
        Hooks are executed in the order they are passed in.
        Hooks do not become part of the agent's assigned functions.
        
        Args:
            func: The base agent function to wrap. Can be any agent's internal function like run, reply_user
            before_hook: List of callable functions to execute before the base function
            after_hook: List of callable functions to execute after the base function
        """
        # Make sure use_function is not wrapped as it will cause infinite loops
        if func == 'use_function':
            raise Exception("use_function cannot be wrapped")
        
        # if functions are not already Function objects, convert all hooks to Function objects. If they are BaseAgent, convert them to Function objects
        before_hook = [hook.to_function(self) if isinstance(hook, BaseAgent) else AsyncFunction(external_fn=hook) for hook in before_hook]
        after_hook = [hook.to_function(self) if isinstance(hook, BaseAgent) else AsyncFunction(external_fn=hook) for hook in after_hook]
                
        async def infer_function_parameters(function: Function):
            input_format = {}
            fn_description = function.fn_description
            matches = re.findall(r'<(.*?)>', fn_description)
            
            # do up an output format dictionary to use to get LLM to output exactly based on keys and types needed
            for match in matches:
                if ':' in match:
                    first_part, second_part = match.split(':', 1)
                    input_format[first_part] = f'A suitable value, type: {second_part}'
                else:
                    input_format[match] = 'A suitable value'
                    
            # if there is no input, then do not need LLM to extract out function's input
            if input_format == {}:
                function_params = {}
                    
            else:
                background_info = f"Assigned Task:```\n{self.task}\n```\nSubtasks Completed: ```{self.subtasks_completed}```"
                # Add in memory to the Agent
                rag_info = ''
                for name in self.memory_bank.keys():
                    # Function RAG is done separately in self.query()
                    if name == 'Function': continue
                rag_info += f'Knowledge Reference for {name}: ```{self.memory_bank[name].retrieve(self.task)}```\n'    

                function_params = await self.query(query = f'''{background_info}{rag_info}\n\n```\nEquipped Function Details: ```{str(function)}```\nOutput suitable values for Inputs to Equipped Function to fulfil Current Subtask\nInput fields are: {list(input_format.keys())}''',
                            output_format = input_format,
                            provide_function_list = False)
            return function_params
        
        try:    
            # Get the original base function
            original_func = getattr(self, func)
        except:
            raise Exception(f"Base function: {func} not found in agent {self.agent_name}")
        
        async def wrapped_function(*args, **kwargs):
            # Special handling for run function to ensure task is assigned before hooks
            if func == 'run' and len(args) > 0:
                self.assign_task(args[0], args[1] if len(args) > 1 else '')
            
            # Execute before hooks
            for hook in before_hook:
                self.assign_functions([hook])
                input_params = await infer_function_parameters(hook)
                await self.use_function(hook.fn_name, input_params)
                self.remove_function(hook.fn_name)
                
            # Execute original function
            result = await original_func(*args, **kwargs)
            
            # Execute after hooks
            for hook in after_hook:
                self.assign_functions([hook])
                input_params = await infer_function_parameters(hook)
                await self.use_function(hook.fn_name, input_params)
                self.remove_function(hook.fn_name)
                
            return result
        
        # Replace the original function with wrapped version
        setattr(self, func, wrapped_function)
    
    ## This is for Multi-Agent uses
    def to_function(self, meta_agent):
        ''' Converts the agent to a function so that it can be called by another agent
        The agent will take in an instruction, and output the result after processing'''

        # makes the agent appear as a function that takes in an instruction and outputs the executed instruction
        my_fn = AsyncFunction(fn_name = self.agent_name,
                             fn_description = f'Agent Description: ```{self.agent_description}```\nExecutes the given <instruction>',
                             output_format = {"Agent Output": "Output of instruction"},
                             external_fn = Async_Agent_External_Function(self, meta_agent))
        
        return my_fn
    
    def assign_agents(self, agent_list: list):
        ''' Assigns a list of Agents to the main agent, passing in the meta agent as well '''
        if not isinstance(agent_list, list):
            agent_list = [agent_list]
        self.assign_functions([agent.to_function(self) for agent in agent_list])
        return self
    
    ## Function aliaises
    assign_function = assign_functions
    assign_tool = assign_functions
    assign_tools = assign_functions
    select_tool = select_function
    use_tool = use_function
    assign_agent = assign_agents
    
###########################################
### This is to wrap Agents as Functions ###
###########################################
    
class Base_Agent_External_Function:
    ''' Creates a Function-based version of the agent '''
    def __init__(self, agent: Agent, meta_agent: Agent):
        ''' Retains the instance of the agent as an internal variable '''
        self.agent = agent
        self.meta_agent = meta_agent


class Agent_External_Function(Base_Agent_External_Function):
    ''' Creates a Function-based version of the agent '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def __call__(self, instruction: str):
        ''' Calls the Inner Agent to perform an instruction. The outcome of the agent goes directly into subtasks_completed
        Returns the Inner Agent's reply '''
        # make a deep copy so we do not affect the original agent
        if self.agent.verbose:
            print(f'\n### Start of Inner Agent: {self.agent.agent_name} ###')
        agent_copy = copy.deepcopy(self.agent)
        
        # take the shared variables from the meta agent
        agent_copy.shared_variables = self.meta_agent.shared_variables
        agent_copy.shared_variables['agent'] = agent_copy
        
        # provide the subtasks completed and debug capabilities to the inner agents too
        agent_copy.reset()
        agent_copy.debug = self.meta_agent.debug
        if len(self.meta_agent.subtasks_completed) > 0:
            agent_copy.global_context += f'Related Subtasks Completed: {self.meta_agent.subtasks_completed}'
        agent_copy.subtasks_completed = {}

        output = agent_copy.run(instruction, self.meta_agent.overall_task)
        
        # append result of inner agent to meta agent
        agent_copy.verbose = False
        agent_reply = agent_copy.reply_user()
        
        if self.agent.verbose:
            print(colored(f'###\nReply from {self.agent.agent_name} to {self.meta_agent.agent_name}:\n{agent_reply}\n###\n', 'magenta', attrs = ['bold']))
            print(f'### End of Inner Agent: {self.agent.agent_name} ###\n')
            
        # sets back the original agent in shared variables
        self.meta_agent.shared_variables['agent'] = self.meta_agent
            
        return agent_reply

class Async_Agent_External_Function(Base_Agent_External_Function):
    ''' Creates a Function-based version of the agent '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.agent, AsyncAgent):
            raise TypeError("Expected Async agent but provided with sync Agent")
        

    async def __call__(self, instruction: str):
        ''' Calls the Inner Agent to perform an instruction. The outcome of the agent goes directly into subtasks_completed
        Returns the Inner Agent's reply '''
        # make a deep copy so we do not affect the original agent
        if self.agent.verbose:
            print(f'\n### Start of Inner Agent: {self.agent.agent_name} ###')
        agent_copy = copy.deepcopy(self.agent)
        
        # take the shared variables from the meta agent
        agent_copy.shared_variables = self.meta_agent.shared_variables
        agent_copy.shared_variables['agent'] = agent_copy
        
        # provide the subtasks completed and debug capabilities to the inner agents too
        agent_copy.reset()
        agent_copy.debug = self.meta_agent.debug
        if len(self.meta_agent.subtasks_completed) > 0:
            agent_copy.global_context += f'Related Subtasks Completed: {self.meta_agent.subtasks_completed}'
        agent_copy.subtasks_completed = {}

        output = await agent_copy.run(instruction, self.meta_agent.overall_task)
        
        # append result of inner agent to meta agent
        agent_copy.verbose = False
        agent_reply = await agent_copy.reply_user()
        
        if self.agent.verbose:
            print(colored(f'###\nReply from {self.agent.agent_name} to {self.meta_agent.agent_name}:\n{agent_reply}\n###\n', 'magenta', attrs = ['bold']))
            print(f'### End of Inner Agent: {self.agent.agent_name} ###\n')
            
        # sets back the original agent in shared variables
        self.meta_agent.shared_variables['agent'] = self.meta_agent
            
        return agent_reply
    

    
