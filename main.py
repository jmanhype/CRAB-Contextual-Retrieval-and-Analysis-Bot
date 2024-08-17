# Import necessary modules
import os
import subprocess
from dotenv import load_dotenv
import dspy
from dspy import OpenAI, settings, Signature, InputField, OutputField, ChainOfThought, Module
from transitions import Machine, MachineError
import re
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from pathlib import Path
from typing import List, Optional, Union, Any
import requests
import json
from loguru import logger
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from munch import Munch
import openai
from dspygen.utils.file_tools import data_dir, count_tokens
from dspygen.utils.dspy_tools import init_ol
from langsmith.wrappers import wrap_openai
from langsmith import traceable, Client

# Load environment variables
load_dotenv()

client = wrap_openai(openai.Client(api_key=os.getenv('OPENAI_API_KEY')))

# Initialize OpenAI settings
llm = OpenAI(
    model='gpt-4o',
    api_key=os.environ['OPENAI_API_KEY'],
    max_tokens=2000
)

# Configure DSPy settings to use the language model
settings.configure(lm=llm)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure loguru logger
logger.add("chatbot_system.log", rotation="10 MB", level="ERROR")

default_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')

class GenerateAnswer(Signature):
    """Answer questions with short factoid answers."""
    context = InputField(desc="may contain relevant facts")
    question = InputField()
    answer = OutputField(desc="often between 1 and 5 words")

class IdentifyConcepts(Signature):
    """Identify key concepts needed to answer a question."""
    context = InputField(desc="initial retrieved context")
    question = InputField()
    concepts = OutputField(desc="list of key concepts")

class EvaluateRetrieval(Signature):
    """Evaluate the quality of retrieved documents."""
    context = InputField(desc="retrieved documents")
    score = OutputField(desc="relevance score")

class RAG(Module):
    def __init__(self, directory, num_passages=3, chunk_size=512, max_files=100, mode='standard'):
        super().__init__()
        self.directory = directory
        self.num_passages = num_passages
        self.chunk_size = chunk_size
        self.max_files = max_files
        self.mode = mode  # Mode can be 'standard' or 'web_search'
        self.generate_answer = ChainOfThought(GenerateAnswer)
        self.identify_concepts = ChainOfThought(IdentifyConcepts)
        self.evaluate_retrieval = ChainOfThought(EvaluateRetrieval)
        self.retriever = self._initialize_retriever()
    
    def _initialize_retriever(self):
        collection_name = "chatbot_system"
        persist_directory = data_dir()
        check_for_updates = True
        embed_fn = default_embed_fn
        k = 5
        
        retriever = ChromaDBRetriever(
            directory=self.directory,
            collection_name=collection_name,
            persist_directory=persist_directory,
            check_for_updates=check_for_updates,
            embed_fn=embed_fn,
            k=k
        )
        
        return retriever
    
    def truncate_context(self, context, max_length=4000):
        """Truncate the context to ensure it doesn't exceed the maximum token length."""
        if len(context) > max_length:
            return context[:max_length]
        return context
    
    def clean_context(self, context):
        """Clean the retrieved context to remove unwanted HTML or script content."""
        clean_text = re.sub(r'<script.*?>.*?</script>', '', context, flags=re.DOTALL)
        clean_text = re.sub(r'<.*?>', '', clean_text)
        return clean_text
    
    @traceable
    def retrieve(self, query):
        # Use the ChromaDBRetriever for enhanced retrieval
        relevant_results = self.retriever.forward(query, k=self.num_passages)
        return relevant_results
    
    @traceable
    def evaluate_retrieval_results(self, documents):
        """Evaluate the quality of retrieved documents."""
        score = self.evaluate_retrieval(context=documents).score
        return score
    
    @traceable
    def decompose_then_recompose(self, documents):
        """Refine the retrieved documents by decomposing and recomposing them."""
        knowledge_strips = []
        for doc in documents:
            strips = doc.split('. ')
            for strip in strips:
                if self.evaluate_retrieval_results(strip) > 0.5:
                    knowledge_strips.append(strip)
        return ' '.join(knowledge_strips)
    
    @traceable
    def web_search(self, query):
        """Perform a web search for additional information using Jina AI's Reader API."""
        search_url = f"https://s.jina.ai/{query.replace(' ', '%20')}"
        try:
            response = requests.get(search_url)
            response.raise_for_status()

            # Clean the response text before attempting to decode JSON
            clean_response_text = response.text.strip()
            
            # Ensure the cleaned response is valid JSON
            try:
                results = json.loads(clean_response_text)
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode JSON response: {json_err}")
                logger.debug(f"Response content: {clean_response_text}")
                return []

            # Check if the results are in the expected format
            if isinstance(results, list) and all('content' in result for result in results):
                web_contents = [result['content'] for result in results]
                return web_contents
            else:
                logger.error("Unexpected format of JSON response")
                logger.debug(f"Response content: {clean_response_text}")
                return []

        except requests.RequestException as req_err:
            logger.error(f"Web search failed: {req_err}")
            return []

    # Example usage:
    # rag = RAG(directory=os.getcwd(), mode='web_search')
    # results = rag.web_search("Roman Empire history")
    # print(results)

            logger.error(f"Failed to decode JSON response: {json_err}")
            logger.debug(f"Response content: {response.text}")
            return []

    @traceable
    def __call__(self, question, context=None):
        if self.mode == 'web_search':
            print("Step 1: Web search retrieval")
            # Step 1: Web search retrieval
            web_results = self.web_search(question)
            final_context = " ".join(web_results)
            final_context = self.truncate_context(final_context)
            final_context = self.clean_context(final_context)
            print("Final context prepared from web search")
        else:
            print("Step 1: Initial retrieval")
            # Step 1: Initial retrieval
            if context is None:
                initial_relevant_results = self.retrieve(question)
                initial_relevant_docs = [result['documents'][0] for result in initial_relevant_results]
                initial_context = " ".join(initial_relevant_docs)
            else:
                initial_context = context
            initial_context = self.truncate_context(initial_context)
            initial_context = self.clean_context(initial_context)
            
            print("Initial context retrieved and cleaned")
            
            print("Step 2: Identify key concepts")
            # Step 2: Identify key concepts
            concepts_result = self.identify_concepts(context=initial_context, question=question)
            key_concepts = concepts_result.concepts
            
            print(f"Identified key concepts: {key_concepts}")
            
            print("Step 3: Enhanced retrieval using key concepts")
            # Step 3: Enhanced retrieval using key concepts
            enhanced_relevant_docs = []
            for concept in key_concepts:
                enhanced_relevant_results = self.retrieve(concept)
                enhanced_relevant_docs.extend([result['documents'][0] for result in enhanced_relevant_results])
            enhanced_context = " ".join(enhanced_relevant_docs)
            enhanced_context = self.truncate_context(enhanced_context)
            enhanced_context = self.clean_context(enhanced_context)
            
            print("Enhanced context retrieved and cleaned")
            
            print("Step 4: Evaluate and correct the retrieval results")
            # Step 4: Evaluate and correct the retrieval results
            relevance_score = self.evaluate_retrieval_results(enhanced_context)
            if relevance_score > 0.75:
                final_context = self.decompose_then_recompose(enhanced_context)
            else:
                final_context = enhanced_context
            
            final_context = self.truncate_context(final_context)
            final_context = self.clean_context(final_context)
            
            print("Final context prepared")
        
        print("Step 5: Generate the final answer using context")
        # Step 5: Generate the final answer using context
        result = self.generate_answer(context=final_context, question=question)
        return result



class ChromaDBRetriever:
    def __init__(
        self,
        directory: str,
        collection_name: str,
        persist_directory: str,
        check_for_updates: bool,
        embed_fn,
        k: int
    ):
        self.directory = directory
        self.collection_name = collection_name
        self.k = k
        self.persist_directory = Path(persist_directory)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.embedding_function = embed_fn
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        if check_for_updates:
            self._process_and_store_files()

    def _process_and_store_files(self):
        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    if len(file_content) < 200:
                        continue

                    self.collection.add(
                        documents=[file_content],
                        metadatas=[{"filename": filename}],
                        ids=[filename],
                    )
                    logger.debug(f"Added file: {filename}")

    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        k: Optional[int] = None,
    ) -> list[str]:
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]  # Filter empty queries

        if not queries:
            logger.error("No valid queries provided")
            return []

        try:
            embeddings = self.embedding_function(queries)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

        if not embeddings or not embeddings[0]:
            logger.error("No embeddings generated")
            return []

        k = self.k if k is None else k

        try:
            results = self.collection.query(
                query_embeddings=embeddings,
                n_results=k,
            )
        except Exception as e:
            logger.error(f"Error querying the collection: {e}")
            return []

        return results



class GeneralInterpreter(Machine):
    states = ['start', 'thinking', 'acting', 'observing', 'concluded']

    def __init__(self, llm, memory_size=5, mode='standard'):
        super().__init__(states=GeneralInterpreter.states, initial='start')
        self.llm = llm
        self.memory = []
        self.memory_size = memory_size
        self.tools = {
            "current_directory": self.get_current_directory,
            "list_directory": self.list_directory_contents,
            "change_directory": self.change_directory,
            "read_file": self.read_file
        }
        self.rag = RAG(directory=os.getcwd(), mode=mode)
        self.mode = mode

        # Add state transitions
        self.add_transition(trigger='think', source='start', dest='thinking')
        self.add_transition(trigger='act', source='thinking', dest='acting')
        self.add_transition(trigger='observe', source='acting', dest='observing')
        self.add_transition(trigger='decide', source='observing', dest='concluded')
        self.add_transition(trigger='restart', source='concluded', dest='start')
        self.add_transition(trigger='restart', source='thinking', dest='start')
        self.add_transition(trigger='restart', source='acting', dest='start')
        self.add_transition(trigger='restart', source='observing', dest='start')
        self.add_transition(trigger='observe', source='thinking', dest='observing')

    def get_current_directory(self):
        """Return the current working directory."""
        try:
            current_dir = os.getcwd()
            return current_dir
        except Exception as e:
            return f"Error: {str(e)}"

    def list_directory_contents(self):
        """Return the contents of the current directory."""
        try:
            contents = os.listdir()
            return contents
        except Exception as e:
            return f"Error: {str(e)}"

    def change_directory(self, directory):
        """Change the current working directory."""
        try:
            os.chdir(directory)
            return f"Changed directory to: {os.getcwd()}"
        except Exception as e:
            return f"Error: {str(e)}"

    def read_file(self, file_path):
        """Read the contents of a file."""
        try:
            with open(file_path, 'r') as file:
                contents = file.read()
            return contents
        except Exception as e:
            return f"Error: {str(e)}"

    def execute_shell_command(self, command):
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout = result.stdout.decode('utf-8')
            stderr = result.stderr.decode('utf-8')
            return stdout, stderr
        except subprocess.CalledProcessError as e:
            return e.stdout.decode('utf-8'), e.stderr.decode('utf-8')

    def tool_function(self, command, *args):
        if command in self.tools:
            result = self.tools[command](*args)
            self.memory.append(f"Tool result for {command}: {result}")
            return result
        else:
            return "Tool function not recognized."

    def set_mode(self, mode):
        """Set the mode for the RAG module."""
        if mode in ['standard', 'web_search']:
            self.mode = mode
            self.rag.mode = mode
            print(f"Mode set to: {mode}")
        else:
            print("Invalid mode. Please choose either 'standard' or 'web_search'.")

    def chat(self):
        print("Hello! How can I help you today?")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            elif user_input.lower().startswith("set mode"):
                mode = user_input.split("set mode")[1].strip()
                self.set_mode(mode)
            elif self.is_tool_command(user_input):
                result = self.handle_tool_command(user_input)
                print(f"AI Response: {result}")
            else:
                self.handle_general_command(user_input)

    def handle_general_command(self, user_input):
        try:
            if self.state == 'start':
                self.trigger('think')
            
            if self.state == 'thinking':
                # Step 1: Retrieve relevant documents from ChromaDB
                retrieved_results = self.rag.retrieve(user_input)
                print(f"DEBUG: Retrieved results: {retrieved_results}")

                # Adjusting extraction logic based on the structure of retrieved_results
                retrieved_docs = [doc for doc in retrieved_results]
                file_names = [meta['filename'] for meta in retrieved_results['metadatas'][0]]

                # Step 2: Pass the retrieved documents to the CRAG system
                context = " ".join(retrieved_docs)
                pred = self.rag(question=user_input, context=context)
                
                print(f"AI Response: {pred.answer}")
                print(f"Files used for generating the answer: {', '.join(file_names)}")
                self.memory.append(pred.answer)
                self.trigger('observe')

            if self.state == 'acting':
                self.trigger('observe')

            if self.state == 'observing':
                self.trigger('decide')

            if self.state == 'concluded':
                self.trigger('restart')
        except MachineError as e:
            print(f"State transition error: {e}")
        except Exception as e:
            print(f"Error: {e}")

    def is_tool_command(self, user_input):
        """Detects if the user input should trigger a tool function call."""
        tool_commands = ['current dir', 'contents of this directory', 'list files', 'list directory', 'current directory',
                         'change directory', 'read file']
        return any(cmd in user_input.lower() for cmd in tool_commands)

    def handle_tool_command(self, user_input):
        """Handles the execution of tool commands based on user input."""
        if 'current dir' in user_input.lower() or 'current directory' in user_input.lower():
            return self.tool_function("current_directory")
        elif 'list files' in user_input.lower() or 'contents of this directory' in user_input.lower() or 'list directory' in user_input.lower():
            return self.tool_function("list_directory")
        elif 'change directory' in user_input.lower():
            directory = user_input.split('change directory')[1].strip()
            return self.tool_function("change_directory", directory)
        elif 'read file' in user_input.lower():
            file_path = user_input.split('read file')[1].strip()
            return self.tool_function("read_file", file_path)
        else:
            return "Tool function not recognized."

# Initialize the chat module
chat_module = GeneralInterpreter(llm, mode='standard')

# Start the chat session
chat_module.chat()
