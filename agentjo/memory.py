from ast import List
import asyncio
import hashlib
import os
import time
from typing import Any
import copy
import pandas as pd

from abc import ABC, abstractmethod

from strictjson import strict_json, strict_json_async, parse_yaml, parse_yaml_async

from agentjo.ranker import AsyncRanker, Ranker
from agentjo.utils import ensure_awaitable, top_k_index, split_text_recursive

###################
## Base Template ##
###################

class MemoryTemplate(ABC):
    """A generic template provided for all memories"""

    @abstractmethod
    def append(self, memory_list, mapper=None):
        """Appends multiple new memories"""
        pass

    # TODO Should this be deleted based on metadata key - value filter
    @abstractmethod
    def remove(self, existing_memory):
        """Removes an existing_memory. existing_memory can be str, or triplet if it is a Knowledge Graph"""
        pass

    @abstractmethod
    def reset(self):
        """Clears all memories"""

    @abstractmethod
    def retrieve(self, task: str):
        """Retrieves some memories according to task"""
        pass
    
    ## Some utility functions
    def read_file(self, filepath):
        if ".xls" in filepath:
            text = pd.read_excel(filepath).to_string()
        elif ".csv" in filepath:
            text = pd.read_csv(filepath).to_string()
        elif ".docx" in filepath:
            text = self.read_docx(filepath)
        elif ".pdf" in filepath:
            text = self.read_pdf(filepath)
        else:
            raise ValueError(
                "File type not spported, supported file types: pdf, docx, csv, xls"
            )

        texts = split_text_recursive(text, max_length=1000, overlap=100, split_by_sentence=False)

        memories = [{"content": text, "filepath": filepath} for text in texts]
        return memories

    def read_pdf(self, filepath):
        import PyPDF2

        # Open the PDF file
        text_list = []
        with open(filepath, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Ensure there's text on the page
                    text_list.append(page_text)
                else:
                    print("No text found on page")
        return "\n".join(text_list)

    def read_docx(self, filepath):
        from docx import Document
        doc = Document(filepath)
        text_list = []
        for para in doc.paragraphs:
            text_list.append(para.text)
        return "\n".join(text_list)

########################
## In-house vector db ##
########################

### BaseMemory is VectorDB that is natively implemented, but not optimised. This will be used for function-based RAG and other kinds of RAG that are not natively text-based. For more optimised memory, check out ChromaDbMemory

class BaseMemory(MemoryTemplate):
    """Retrieves top k memory items based on task. This is an in-house, unoptimised, vector db
    - Inputs:
        - `memory`: List. Default: None. The list containing the memory items
        - `top_k`: Int. Default: 3. The number of memory list items to retrieve
        - `mapper`: Function. Maps the memory item to another form for comparison by ranker or LLM. Default: `lambda x: x`
            - Example mapping: `lambda x: x.fn_description` (If x is a Class and the string you want to compare for similarity is the fn_description attribute of that class)
        - `approach`: str. Either `retrieve_by_ranker` or `retrieve_by_llm` to retrieve memory items
            - Ranker is faster and cheaper as it compares via embeddings, but are inferior to LLM-based methods for contextual information
        - `llm`: Function. The llm to use for llm retriever
        - `llm_parser`: Function. The llm parser. Either parse_yaml or strict_json
        - `llm_parser_async`: Function. The llm parser for async mode.  Either parse_yaml_async or strict_json_async
        - `retrieve_fn`: Default: None. Takes in task and outputs top_k similar memories in a list. Does away with the Ranker() altogether
        - `ranker`: `Ranker`. The Ranker which defines a similarity score between a query and a key. Default: OpenAI `text-embedding-3-small` model.
            - Can be replaced with a function which returns similarity score from 0 to 1 when given a query and key
    """

    def __init__(
        self,
        memory: list = None,
        top_k: int = 3,
        mapper=lambda x: x,
        approach="retrieve_by_ranker",
        llm=None,
        retrieve_fn=None,
        llm_parser = parse_yaml,
        llm_parser_async = parse_yaml_async,
        ranker=None,
    ):
        if memory is None:
            self.memory = []
        else:
            self.memory = memory
        self.top_k = top_k
        self.mapper = mapper
        self.approach = approach
        self.ranker = ranker
        self.retrieve_fn = retrieve_fn
        self.llm_parser = llm_parser
        self.llm_parser_async = llm_parser_async
        self.llm = llm

    def add_file(self, filepath):
        memories = self.read_file(filepath)
        texts = [memory["content"] for memory in memories]
        self.append(texts)

    def append(self, memory_list, mapper=None):
        """Adds a list of memories"""
        if not isinstance(memory_list, list):
            memory_list = [memory_list]
        self.memory.extend(memory_list)

    def remove(self, memory_to_remove):
        """Removes a memory"""
        self.memory.remove(memory_to_remove)

    def reset(self):
        """Clears all memory"""
        self.memory = []

    def isempty(self) -> bool:
        """Returns whether or not the memory is empty"""
        return not self.memory

class Memory(BaseMemory):
    """Retrieves top k memory items based on task
    - Inputs:
        - `memory`: List. Default: Empty List. The list containing the memory items
        - `top_k`: Int. Default: 3. The number of memory list items to retrieve
        - `mapper`: Function. Maps the memory item to another form for comparison by ranker or LLM. Default: `lambda x: x`
            - Example mapping: `lambda x: x.fn_description` (If x is a Class and the string you want to compare for similarity is the fn_description attribute of that class)
        - `approach`: str. Either `retrieve_by_ranker` or `retrieve_by_llm` to retrieve memory items
            - Ranker is faster and cheaper as it compares via embeddings, but are inferior to LLM-based methods for contextual information
        - `llm`: Function. The llm to use for the llm retriever
        - `retrieve_fn`: Default: None. Takes in task and outputs top_k similar memories in a list
        - `ranker`: `Ranker`. The Ranker which defines a similarity score between a query and a key. Default: OpenAI `text-embedding-3-small` model.
            - Can be replaced with a function which returns similarity score from 0 to 1 when given a query and key
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ranker is None:
            self.ranker = (
                Ranker()
            )  # Assuming Ranker needs to be initialized if not provided

    def retrieve(self, task: str) -> list:
        """Performs retrieval of top_k similar memories according to approach stated"""
        # if you have your own vector search function, implement it in retrieve_fn. Takes in a task and outputs the top-k results
        if self.retrieve_fn is not None:
            return self.retrieve_fn(task)
        else:
            if self.approach == "retrieve_by_ranker":
                return self.retrieve_by_ranker(task)
            else:
                return self.retrieve_by_llm(task)

    def retrieve_by_ranker(self, task: str) -> list:
        """Performs retrieval of top_k similar memories
        Returns the memory list items corresponding to top_k matches"""
        # if there is no need to filter because top_k is already more or equal to memory size, just return memory
        if self.top_k >= len(self.memory):
            return copy.deepcopy(self.memory)

        # otherwise, perform filtering
        else:
            memory_score = [
                self.ranker(self.mapper(memory_chunk), task)
                for memory_chunk in self.memory
            ]
            top_k_indices = top_k_index(memory_score, self.top_k)
            return [self.memory[index] for index in top_k_indices]

    def retrieve_by_llm(self, task: str) -> list:
        """Performs retrieval via LLMs
        Returns the key list as well as the value list"""
        res = self.llm_parser(
            f'You are to output the top {self.top_k} most similar list items in Memory relevant to this: ```{task}```\nMemory: {[f"{i}. {self.mapper(mem)}" for i, mem in enumerate(self.memory)]}',
            "",
            output_format={
                f"top_{self.top_k}_list": f"Indices of top {self.top_k} most similar list items in Memory, type: list[int]"
            },
            llm=self.llm,
        )
        top_k_indices = res[f"top_{self.top_k}_list"]
        return [self.memory[index] for index in top_k_indices]


class AsyncMemory(BaseMemory):
    """Retrieves top k memory items based on task
    - Inputs:
        - `memory`: List. Default: Empty List. The list containing the memory items
        - `top_k`: Int. Default: 3. The number of memory list items to retrieve
        - `mapper`: Function. Maps the memory item to another form for comparison by ranker or LLM. Default: `lambda x: x`
            - Example mapping: `lambda x: x.fn_description` (If x is a Class and the string you want to compare for similarity is the fn_description attribute of that class)
        - `approach`: str. Either `retrieve_by_ranker` or `retrieve_by_llm` to retrieve memory items
            - Ranker is faster and cheaper as it compares via embeddings, but are inferior to LLM-based methods for contextual information
        - `llm`: Function. The llm to use for llm retriever
        - `retrieve_fn`: Default: None. Takes in task and outputs top_k similar memories in a list
        - `ranker`: `Ranker`. The Ranker which defines a similarity score between a query and a key. Default: OpenAI `text-embedding-3-small` model.
            - Can be replaced with a function which returns similarity score from 0 to 1 when given a query and key
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ranker is None:
            self.ranker = (
                AsyncRanker()
            )  # Assuming Ranker needs to be initialized if not provided
        if not isinstance(self.ranker, AsyncRanker):
            raise Exception("Sync Ranker not allowed in AsyncMemory")
        ensure_awaitable(self.retrieve_fn, "retrieve_fn")
        ensure_awaitable(self.llm, "llm")
        ensure_awaitable(self.llm_parser_async, "llm parser")

    async def retrieve(self, task: str) -> list:
        """Performs retrieval of top_k similar memories according to approach stated"""
        # if you have your own vector search function, implement it in retrieve_fn. Takes in a task and outputs the top-k results
        if self.retrieve_fn is not None:
            return await self.retrieve_fn(task)
        else:
            if self.approach == "retrieve_by_ranker":
                return await self.retrieve_by_ranker(task)
            else:
                return await self.retrieve_by_llm(task)

    async def retrieve_by_ranker(self, task: str) -> list:
        """Performs retrieval of top_k similar memories
        Returns the memory list items corresponding to top_k matches"""
        # if there is no need to filter because top_k is already more or equal to memory size, just return memory
        if self.top_k >= len(self.memory):
            return copy.deepcopy(self.memory)

        # otherwise, perform filtering
        else:
            tasks = [
                self.ranker(self.mapper(memory_chunk), task)
                for memory_chunk in self.memory
            ]
            memory_score = await asyncio.gather(*tasks)
            top_k_indices = top_k_index(memory_score, self.top_k)
            return [self.memory[index] for index in top_k_indices]

    async def retrieve_by_llm(self, task: str) -> list:
        """Performs retrieval via LLMs
        Returns the key list as well as the value list"""
        res = await self.llm_parser_async(
            f'You are to output the top {self.top_k} most similar list items in Memory relevant to this: ```{task}```\nMemory: {[f"{i}. {self.mapper(mem)}" for i, mem in enumerate(self.memory)]}',
            "",
            output_format={
                f"top_{self.top_k}_list": f"Indices of top {self.top_k} most similar list items in Memory, type: list[int]"
            },
            llm=self.llm,
        )
        top_k_indices = res[f"top_{self.top_k}_list"]
        return [self.memory[index] for index in top_k_indices]