# CRAB: Contextual Retrieval and Analysis Bot

CRAB (Contextual Retrieval and Analysis Bot) is a sophisticated AI-powered conversational system that combines Retrieval-Augmented Generation (RAG), state machine-based conversation flow, and various AI tools to provide intelligent responses and perform file system operations.

## Features

- RAG-based question answering using local documents and optional web search
- State machine-driven conversation flow
- File system operations (list directory, change directory, read files)
- Flexible mode switching between standard and web search
- Integration with ChromaDB for efficient document retrieval
- Sentence embedding for semantic search
- Logging and error handling
- Integration with LangSmith for tracing and monitoring

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/CRAB.git
   cd CRAB
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

To start CRAB, run:

```
python crab_main.py
```

Once CRAB is running, you can:

- Ask questions and receive RAG-enhanced responses
- Use file system commands like "list files", "change directory", "read file"
- Switch modes between standard and web search using "set mode standard" or "set mode web_search"
- Type "exit" to end the chat session

## Project Structure

- `crab_main.py`: Main script containing the CRAB implementation
- `requirements.txt`: List of Python dependencies
- `.env`: Environment variables (create this file and add your API keys)

## Customization

- Modify the `RAG` class to adjust retrieval and generation strategies
- Extend the `GeneralInterpreter` class to add new tools or conversation states
- Adjust the `ChromaDBRetriever` class to change document indexing and retrieval methods

## Contributing

Contributions to CRAB are welcome! Please fork the repository, create a feature branch, and submit a pull request for review.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Disclaimer

CRAB is an experimental AI system. Always review and validate its outputs, especially for critical applications. The system may occasionally produce incorrect or biased information.
