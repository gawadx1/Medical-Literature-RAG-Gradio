# Medical Literature RAG System with Gradio

This project implements a Retrieval-Augmented Generation (RAG) system designed to summarize medical literature from PDF documents and provide evidence-based answers to user queries. It features a user-friendly web interface built with Gradio, allowing users to upload PDF files, process them, and generate various types of summaries (comprehensive, clinical, research, executive).

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Summary Types](#summary-types)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction
In the vast and ever-growing field of medical literature, quickly extracting relevant information and synthesizing it into concise, accurate summaries is crucial for healthcare professionals, researchers, and decision-makers. This RAG system addresses this challenge by combining the power of large language models (LLMs) with a robust information retrieval mechanism. By processing PDF documents, chunking their content, and creating vector embeddings, the system can efficiently find the most pertinent information to answer specific questions or generate summaries, ensuring that the AI's responses are grounded in the provided medical texts.




## Features

- **PDF Content Extraction**: Robust extraction of text content and metadata from medical PDF documents using both PyMuPDF and PyPDF2 for enhanced reliability.
- **Intelligent Section Parsing**: Identifies and extracts key medical sections within PDFs (e.g., Abstract, Introduction, Methods, Results, Discussion, Conclusion) to provide structured access to information.
- **Content Chunking with Overlap**: Breaks down large PDF content into smaller, overlapping chunks to optimize retrieval accuracy and context preservation for the language model.
- **FAISS Vector Store**: Utilizes FAISS (Facebook AI Similarity Search) for efficient storage and rapid similarity search of document embeddings, enabling fast retrieval of relevant content.
- **Sentence Embeddings**: Leverages `SentenceTransformer` (specifically `all-MiniLM-L6-v2`) to convert text chunks and queries into high-dimensional vector embeddings.
- **Groq-powered RAG**: Integrates with Groq's API (using `meta-llama/llama-4-scout-17b-16e-instruct`) to perform Retrieval-Augmented Generation, ensuring summaries and answers are accurate and contextually relevant.
- **Multiple Summary Types**: Offers various summarization modes tailored for different needs:
    - **Comprehensive**: A general overview covering main findings, methodology, and conclusions.
    - **Clinical Focus**: Emphasizes clinical relevance, treatment implications, patient outcomes, and practical recommendations.
    - **Research Focus**: Concentrates on research questions, methodology, statistical results, limitations, and future directions.
    - **Executive Summary**: Provides concise key points, clinical impact, action items, and next steps for decision-makers.
- **Gradio Web Interface**: Provides an intuitive and interactive web application for:
    - Uploading multiple PDF files.
    - Viewing processing status and document overviews.
    - Inputting natural language queries.
    - Selecting desired summary types.
    - Displaying AI-generated summaries and their original sources.
- **Temporary File Handling**: Manages temporary PDF files securely during processing, ensuring proper cleanup.




## Installation

To set up the Medical Literature RAG System, follow these steps:

1.  **Clone the repository** (if applicable, assuming this code is part of a larger repository):

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages**:

    ```bash
    pip install groq sentence-transformers faiss-cpu PyPDF2 PyMuPDF numpy gradio
    ```

4.  **Obtain a Groq API Key**:
    -   Sign up or log in to the [Groq Cloud](https://console.groq.com/keys).
    -   Generate a new API key.
    -   Replace `GROQ_API_KEY = "gsk_YOUR_API_KEY_HERE"` in the `Medical_Literature_RAG_System&Gradio.ipynb` file with your actual API key. For security, it's recommended to use environment variables for API keys in production environments.




## Usage

To run the Medical Literature RAG System and access its Gradio web interface, execute the Jupyter notebook:

```bash
jupyter notebook Medical_Literature_RAG_System&Gradio.ipynb
```

Alternatively, you can convert the notebook to a Python script and run it:

```bash
jupyter nbconvert --to script Medical_Literature_RAG_System&Gradio.ipynb
python Medical_Literature_RAG_System&Gradio.py
```

Once the application is running, open the displayed local URL (e.g., `http://127.0.0.1:7860` or a public Gradio share link if `share=True` is enabled) in your web browser.

### How to Use the Interface:

1.  **Upload Medical PDFs**: In the "Upload Medical PDFs" section, click on the file upload area or drag and drop your PDF documents. You can upload multiple files at once.
2.  **Process PDFs**: Click the "🔄 Process PDFs" button. The "Processing Status" textbox will show the progress and outcome of the PDF extraction and processing. The "Document Overview" section will update with details of the loaded documents.
3.  **Query Your Documents**: In the "Query Your Documents" section:
    -   Enter your question in the "Your Question" textbox. Leave it empty for a general summary of the loaded documents.
    -   Select the desired "Summary Type" from the dropdown menu (Comprehensive, Clinical Focus, Research Focus, Executive Summary).
4.  **Generate Summary**: Click the "✨ Generate Summary" button. The system will retrieve relevant information and generate a summary based on your query and selected summary type.
5.  **View Results**: The generated summary will appear in the "Summary" section, and the sources (filenames, titles, pages, and sections) will be listed in the "Sources & References" section.
6.  **Clear All**: To clear all loaded documents and reset the interface, click the "🗑️ Clear All" button.




## Project Structure

The project consists primarily of a single Jupyter Notebook file, `Medical_Literature_RAG_System&Gradio.ipynb`, which contains all the code for the RAG system and the Gradio interface. When executed, it sets up the environment, defines the core classes and functions, and launches the web application.

```
.
├── Medical_Literature_RAG_System&Gradio.ipynb
└── README.md
```




## Core Components

The system is built around several key Python classes and functions:

### `PDFDocument` (dataclass)

A dataclass representing a single PDF medical document. It stores:
-   `filename`: Original filename of the PDF.
-   `title`: Extracted title of the document.
-   `content`: Full text content extracted from the PDF.
-   `sections`: A dictionary mapping medical section names (e.g., 'abstract', 'methods') to their extracted text content.
-   `metadata`: Any metadata extracted from the PDF (e.g., author, creation date).
-   `page_count`: Total number of pages in the PDF.
-   `embedding`: (Not directly used in the current RAG flow for full document, but can be extended for document-level embeddings).
-   `chunks`: A list of dictionaries, each representing an overlapping text chunk from the document, along with its `chunk_id`, `start_word`, and `word_count`.

### `PDFProcessor`

This class is responsible for handling the extraction and initial processing of PDF files.

-   **`__init__`**: Initializes with a predefined list of common medical section names to look for.
-   **`extract_pdf_content(pdf_path: str) -> PDFDocument`**: The main method to extract content from a given PDF path. It attempts extraction using `PyMuPDF` first for better accuracy with complex layouts, falling back to `PyPDF2` if `PyMuPDF` fails. It also extracts a potential title, parses medical sections, and creates content chunks.
-   **`_extract_with_pymupdf(pdf_path: str) -> tuple`**: Internal method to extract text and metadata using the `fitz` (PyMuPDF) library.
-   **`_extract_with_pypdf2(pdf_path: str) -> tuple`**: Internal fallback method to extract text and metadata using the `PyPDF2` library.
-   **`_extract_title(content: str) -> Optional[str]`**: Attempts to identify and extract the main title from the beginning of the PDF content.
-   **`_parse_medical_sections(content: str) -> Dict[str, str]`**: Iterates through the content to find and extract text corresponding to predefined medical sections (e.g., 'introduction', 'results').
-   **`_create_content_chunks(content: str, filename: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]`**: Divides the document's content into smaller, overlapping chunks. This is crucial for RAG, as it allows the system to retrieve specific, relevant passages rather than entire documents.

### `PDFVectorStore`

Manages the creation and searching of a vector index for the PDF content chunks.

-   **`__init__(self, embedding_model: str = "all-MiniLM-L6-v2")`**: Initializes the `SentenceTransformer` model (defaulting to `all-MiniLM-L6-v2`) for generating embeddings and sets up internal lists to store documents and chunks. It also initializes a `FAISS` index.
-   **`add_pdf_documents(self, documents: List[PDFDocument])`**: Takes a list of `PDFDocument` objects, extracts their chunks, generates embeddings for each chunk using the `SentenceTransformer`, and adds these embeddings to the `FAISS` index. Embeddings are L2-normalized for cosine similarity search.
-   **`search(self, query: str, k: int = 5) -> List[Dict[str, Any]]`**: Performs a similarity search on the `FAISS` index using an encoded query. It returns the top `k` most relevant content chunks, along with their associated document information and relevance scores.

### `PDFMedicalRAG`

The central class orchestrating the RAG process, combining PDF processing, vector storage, and LLM interaction.

-   **`__init__(self, groq_api_key: str)`**: Initializes the Groq client with the provided API key, and creates instances of `PDFProcessor` and `PDFVectorStore`.
-   **`load_pdf(self, pdf_path: str) -> PDFDocument`**: Loads and processes a single PDF file. It uses the `PDFProcessor` to extract content and then adds the document's chunks to the `PDFVectorStore`.
-   **`summarize_pdf(self, query: str = None, summary_type: str = "comprehensive") -> Dict[str, Any]`**: The main function for generating summaries. It takes a user query (or uses a default if none is provided) and a `summary_type`. It searches the `PDFVectorStore` for relevant content, then uses the retrieved content to prompt the Groq LLM to generate a summary.
-   **`_generate_pdf_summary(self, query: str, search_results: List[Dict], summary_type: str) -> str`**: Constructs the prompt for the Groq LLM, incorporating the user's query and the retrieved relevant content. It uses different prompt templates based on the `summary_type` (comprehensive, clinical, research, executive) to guide the LLM's response.
-   **`get_document_info(self) -> List[Dict[str, Any]]`**: Returns a list of dictionaries containing information about all currently loaded documents.
-   **`clear_documents(self)`**: Resets the RAG system by clearing all loaded documents and re-initializing the vector store.

### Gradio Interface Functions

-   **`upload_and_process_pdf(pdf_files)`**: Handles the file upload from the Gradio interface. It saves temporary copies of the uploaded PDFs, processes them using `rag_system.load_pdf`, and updates the Gradio interface with processing status and document overviews.
-   **`generate_summary(query, summary_type, progress)`**: Triggers the summary generation process via `rag_system.summarize_pdf`. It includes progress tracking for the Gradio interface and formats the output summary and sources for display.
-   **`clear_all()`**: Resets the Gradio interface and clears all loaded documents from the `rag_system`.
-   **`create_interface()`**: Defines the layout and components of the Gradio web application, including file uploaders, textboxes, dropdowns, and buttons, and sets up event handlers for user interactions.
-   **`main()`**: The entry point of the script, which checks for required packages, initializes the Gradio interface, and launches the web application.




## Summary Types

The system offers four distinct summary types, each tailored to different information needs:

1.  **Comprehensive Summary**:
    -   **Purpose**: Provides a holistic overview of the medical literature.
    -   **Key Information**: Document Overview (type of literature), Key Findings (main research outcomes, clinical findings), Methodology (research approaches, study designs), Clinical Significance (practical applications), Evidence Quality (strength and reliability), and Conclusions (main takeaways).
    -   **Audience**: General healthcare professionals, researchers, or anyone needing a complete understanding.

2.  **Clinical Focus Summary**:
    -   **Purpose**: Highlights information most relevant to patient care and clinical practice.
    -   **Key Information**: Clinical Relevance (direct applications), Treatment Implications (impact on practice), Patient Outcomes (impact on health), Evidence Level (quality of clinical evidence), Practical Recommendations (actionable guidance), and Safety Considerations (warnings, contraindications).
    -   **Audience**: Clinicians, medical practitioners, and healthcare providers.

3.  **Research Focus Summary**:
    -   **Purpose**: Emphasizes the scientific and methodological aspects of the research.
    -   **Key Information**: Research Questions (aims of study), Methodology (study design, sample size, analytical methods), Statistical Results (quantitative findings, significance), Limitations (study biases), Implications (advances in knowledge), and Future Directions (recommended follow-up research).
    -   **Audience**: Medical researchers, academics, and statisticians.

4.  **Executive Summary**:
    -   **Purpose**: Delivers concise, high-level key points for quick decision-making.
    -   **Key Information**: Key Points (most important findings in bullet points), Clinical Impact (bottom-line implications), Action Items (what should be done), Risk/Benefit (important considerations), Timeline (time-sensitive info), and Next Steps (recommended follow-up actions).
    -   **Audience**: Decision-makers, administrators, and busy professionals.




## Dependencies

The project relies on the following Python libraries:

-   `groq`: For interacting with the Groq API to generate summaries.
-   `sentence-transformers`: For creating sentence embeddings from text.
-   `faiss-cpu`: For efficient similarity search and vector indexing.
-   `PyPDF2`: A pure-Python PDF library for extracting text and metadata.
-   `PyMuPDF` (fitz): A high-performance PDF library for robust PDF content extraction.
-   `numpy`: For numerical operations, especially with embeddings.
-   `gradio`: For building the interactive web user interface.
-   `os`, `json`, `re`, `dataclasses`, `datetime`, `tempfile`, `shutil`: Standard Python libraries for file operations, data handling, regular expressions, and temporary file management.




## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature` or `bugfix/YourBugfix`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.




## License

This project is licensed under the MIT License - see the LICENSE file for details.



