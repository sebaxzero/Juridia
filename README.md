# ChainPDF
A Query-Answer chatbot for PDFs using local Large language and Embeddings models. Please read this readme fully before using.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [GPU](#gpu)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)


## Features

- Conversational Capabilities with PDFs:

Utilizes the conversational retrieval chain from "langchain" to retrieve information from PDFs and respond to user queries while keeping a memory of the interactions and sources of the information retrieved.

- Fully Local:

Apart from downloading the necessary libraries, this chatbot can be used offline, ensuring privacy and accessibility even without an internet connection.

- Fully Customizable:

Easily customize various parameters, prompt templates, models, and more to tailor the chatbot's behavior to your specific needs.

- GPU Support:

Leverage your GPU to accelerate the text-to-embedding process and text generation (applicable when using a local llm backend).

- Streamlit GUI:

An example provided using a Streamlit GUI for a user-friendly interface.

- Multiple LLM Backend Choices:

Choose between using the textgen webui API, OpenAI API (compatible server API), or directly use llama.cpp for your preferred language model backend.

- Multiple Embeddings Models Choices:

Select from Sentence Transformer or Instructor Transformer embedding models for different use cases.

## Installation

### In Windows:

- Requirements:
  - [Python 3.10+](https://www.python.org/downloads/): Ensure you have the latest Python version installed.
  - [Git](https://git-scm.com/download/win): Install Git and remember to select "Add to PATH" during installation for both Python and Git.

1. Clone the Repository:
    ```
    git clone https://github.com/sebaxzero/ChainPDF
    ```
2. Navigate to the Project Directory:
    ```
    cd ChainPDF
    ```
3. Create a Virtual Environment (Optional but recommended):
   - You can create a Python virtual environment using the following command:
        ```
        python -m venv "venv"
        ```
   - Activate the created venv using:
        ```
        call "venv\Scripts\activate.bat"
        ```
- Optionally, you can use the provided `create_conda_env.bat` file to create a Miniconda environment, which will also install the requirements, so you can skip step 4. This will take some time (this requires a CUDA compatible GPU).

4. Install Requirements:
    ```
    pip install -r requirements.txt
    ```

### OpenAI API and LLama.cpp

These libraries are not included in the `requirements.txt`.
   - To install the OpenAI library, use the following command:
        ```
        pip install openai
        ```
   - To install llama.cpp, use the following command:
        ```
        pip install llama-cpp-python
        ```
**_NOTE:_** This will enable llama.cpp to be used on the CPU. For GPU acceleration, see the next section.

## GPU

For GPU usage, I recommend using the provided `create_conda_env.bat` file. This will create a Miniconda virtual environment, install torch compiled with CUDA alongside the requirements, this will accelerate the text-to-embedding process which can take hours on cpu with large amount of data.

To install llama.cpp with [GPU acceleration](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal), use the following command:
```
SET LLAMA_CLBLAST=1 && SET CMAKE_ARGS="-DLLAMA_CLBLAST=on" && SET FORCE_CMAKE=1 && python -m pip install llama-cpp-python
```

If you are using [TextGen Webui](https://github.com/oobabooga/text-generation-webui), installed with the [one-click-installers](https://github.com/oobabooga/one-click-installers), you can use the same conda environment to run the code, so there's no need to download another instance of CUDA libraries, which are big. To do this, open `cmd_windows.bat` and follow steps 1, 2, and 4.

## Usage

To run the application, activate the environment using your selected method of installation, then use the following command:
```
streamlit run st_interface_en.py
```

## How it works

uses an [Embedding model](https://python.langchain.com/docs/modules/data_connection/text_embedding/) to create embeddings from the provided document and stores it in a vectorStore ([see example](https://python.langchain.com/docs/modules/data_connection/vectorstores/)). When a user asks a question, the chatbot passes the question to a [retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/), which retrieves relevant data from the vectorstore. The retrieved information is then passed to the [qa chain](https://python.langchain.com/docs/modules/chains/popular/chat_vector_db), obtaining a response from the selected llm backend alongside the retrieved pieces of information.

### Code Example
This is a simplified example of the code:
```python
from src.main import Memory, LLM, VectorStore, Chain

memory = Memory.get() # memory variable to save chat history
llm = LLM.get(llm='TextGen') # llm variable
retriever = VectorStore.retriever(k=3, db=VectorStore.get(name='example')) 

while True:
    prompt = input('QUESTION: ')
    answer, source_documents = Chain.query(prompt=prompt, chain=Chain.Get(llm=llm, retriever=retriever, memory=memory))
    print("ANSWER:\n", answer, "\n\n")
```

This code reads any document (.pdf or .txt) placed in the `./Documents` directory, saving the generated VectorStore in `./Sessions/example/Index`. It then retrieves `3` relevant chunks of information passed to the chain, which is sent to the `Texgen webui API` to obtain an answer.

## Contributing

Contributions are welcome! Feel free to contribute to this project and make it even better.


## License

This project is licensed under the MIT License.
