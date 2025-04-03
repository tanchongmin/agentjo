import ast
import heapq
import inspect
import re
    
def ensure_awaitable(func, name):
    """ Utility function to check if the function is an awaitable coroutine function """
    if func is not None and not inspect.iscoroutinefunction(func):
        raise TypeError(f"{name} must be an awaitable coroutine function")
    
### Helper Functions
def top_k_index(lst, k):
    ''' Given a list lst, find the top k indices corresponding to the top k values '''
    indexed_lst = list(enumerate(lst))
    top_k_values_with_indices = heapq.nlargest(k, indexed_lst, key=lambda x: x[1])
    top_k_indices = [index for index, _ in top_k_values_with_indices]
    return top_k_indices

def split_text_recursive(text, max_length=1000, overlap=100, split_by_sentence=False):
    ''' Splits a text into chunks recursively
    Inputs:
    text (str): Original text
    max_length (int): Max length in characters for each chunk
    overlap (int): Character overlap between chunks
    split_by_sentence (bool): Whether or not to preserve punctuation and split at the end of sentences
    '''
    
    if len(text) <= max_length:
        return [text]  # Base case: text fits within the limit

    if split_by_sentence:
        # Use regex to split into sentences while preserving punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from the previous chunk
                current_chunk = " ".join(current_chunk.split()[-(overlap//10):]) + " " + sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # Default splitting method (word-aware)
    split_point = max_length
    while split_point > 0 and text[split_point] not in " .,;!?":
        split_point -= 1

    if split_point == 0:
        split_point = max_length  # Force split if no good breakpoint is found

    chunk = text[:split_point].strip()
    next_start = max(0, split_point - overlap)  # Ensure overlap
    remaining_text = text[next_start:].strip()

    return [chunk] + split_text_recursive(remaining_text, max_length, overlap, split_by_sentence)

### Utility function to run and debug code ###

def python_generator_tool(shared_variables, instruction: str):
    '''Generate code only based on instruction without any additional context.
Ensure that you define all variables and list out all imports.
You can only import the following modules: math, numpy, random, datetime, re, matplotlib, pandas, plotly
Do not define any new functions
You are able to use all Equipped Functions except use_llm and end_task
The output of Equipped Function will be in a dictionary format
Ensure all required output are in print()'''
    
    agent = shared_variables['agent']
    
    return agent.llm_parser(f'''Generate code based only on ```{instruction}``` without additional context.
Ensure that you define all variables and list out all imports.
You can only import the following modules: math, numpy, random, datetime, re, matplotlib, pandas, plotly
Do not define any new functions
You are able to use the following Equipped Functions: 
```{agent.list_functions(
fn_list = [agent.function_map[function_name] for function_name in agent.function_map if function_name not in ['use_llm', 'end_task']])}
```
The output of Equipped Function will be in a dictionary format
You must use Equipped Functions whenever possible
Ensure all required output are in print()''',
                       '',
                                     output_format = {'Code': 'Generated code, type: code'},
                      llm = agent.llm)

def python_run_tool(shared_variables, code_snippet: str) -> str:
    '''Runs code_snippet and outputs the result of all print statements'''
    import sys
    import io
    import math
    import numpy
    import random
    import datetime
    import re
    import matplotlib
    import pandas
    import plotly
    
    agent = shared_variables['agent']
    
    # wrap external functions to pass in shared_variables as well
    def external_function_wrapper(function_name, shared_variables):
        def external_function(**function_params):
            return agent.function_map[function_name](shared_variables=shared_variables, **function_params)
        return external_function
    
    # Capture the output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        # Safe environment to execute the user code
        allowed_globals = {
            '__builtins__': {
                'print': print,
                'range': range,
                'len': len,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'any': any,
                'all': all,
                'sorted': sorted,
                'zip': zip,
                'map': map,
                'filter': filter,
                '__import__': __import__,
                'math': math,  # Allow access to the math module
                'datetime': datetime, # Allow access to datetime module
                'random': random, # Allow access to random module
                'numpy': numpy, # Allow access to numpy module
                're': re,
                'matplotlib': matplotlib,
                'pandas': pandas,
                'plotly': plotly
            }
        }
        
        # add in equipped functions one by one
        for function_name in agent.function_map:
            if function_name not in ['use_llm', 'end_task', 'python_generate_and_run_code_tool']:
                allowed_globals['__builtins__'][function_name] = external_function_wrapper(function_name, shared_variables)

        safe_locals = {}

        exec(code_snippet, allowed_globals, safe_locals)
        output = sys.stdout.getvalue()
    except Exception as e:
        output = f"Error: {e}"
    finally:
        # Restore the original stdout
        sys.stdout = old_stdout

    return output

def python_debug_tool(shared_variables, instruction: str, python_code: str, error_msg: str) -> str:
    '''Takes in intended instruction, current python_code and current error_msg, and outputs corrected code'''

    agent = shared_variables['agent']
    
    return agent.llm_parser('Debugs Python Code and returns corrected code.',
f'''Instruction: {instruction}
Current Code: {python_code}
Error Message: {error_msg}''',
    output_format = {'Thoughts': 'How to correct code', 'Corrected Code': 'type: code'}, 
                             fn_name = 'python_debug_tool', 
                             llm = agent.llm)

# Tool to generate and run code
def python_generate_and_run_code_tool(shared_variables, instruction: str) -> str:
    ''' Generates and runs code based on instruction. 
You can only import the following modules: math, numpy, random, datetime, re, matplotlib, pandas, plotly
You can use all Equipped Functions except use_llm and end_task. 
Returns 1) the result of all print statements in code, or error messages, and 2) the code '''
    # from termcolor import colored
    
    # Append context to tool
    if shared_variables and 'agent' in shared_variables:
        instruction = f"Context: {shared_variables['agent'].overall_task}\nPrevious Subtasks: {shared_variables['agent'].subtasks_completed}\nInstruction: {instruction}"
    # Generate Code
    python_code = python_generator_tool(shared_variables, instruction)['Code']
    
    # Run and Debug Code
    for _ in range(3):
        output = python_run_tool(shared_variables, python_code)

        if output[:5] == "Error":
            debugged_code = python_debug_tool(shared_variables, instruction, python_code, output)
            python_code = debugged_code['Corrected Code']
        else:
            break
            
    return output, python_code

