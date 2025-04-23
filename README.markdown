# RAG-based AI Assistant for TAC Engineers

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system designed to assist **Technical Assistance Center (TAC)** engineers at a networking services company. The AI assistant leverages a PDF-based knowledge base (e.g., technical documentation for network switches like MES5448 and MES7048) to provide accurate and contextually relevant answers to technical queries. By combining semantic search with a large language model (LLM), the system enables engineers to respond to customer inquiries efficiently and precisely.

The project uses **PyMuPDF** for PDF processing, **SentenceTransformers** for embedding generation, and **Google's Gemma-7B** model for text generation, optimized with 4-bit quantization and Flash Attention for efficient inference.

## Features

- **PDF Text Extraction**: Extracts and preprocesses text from technical PDFs, splitting it into manageable chunks.
- **Semantic Search**: Uses `all-mpnet-base-v2` to create embeddings and retrieve the most relevant text chunks based on user queries.
- **Contextual Response Generation**: Integrates retrieved context with the Gemma-7B model to generate detailed and accurate answers.
- **Optimized Inference**: Employs 4-bit quantization and Flash Attention to run the model on resource-constrained hardware.
- **Structured Prompt Engineering**: Utilizes well-designed prompts with examples to ensure high-quality, task-specific responses.

## Project Structure

```
├── rag_copy.py                # Main script for the RAG pipeline
├── knowledge_base.pdf         # Sample PDF (not included; placeholder for technical documentation)
├── text_chunks_and_embeddings_df.csv # Stored embeddings for text chunks
├── README.md                 # Project documentation
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for embedding and LLM inference)
-- mine : NVIDIA GeForce RTX 3090
- Required libraries (see below)

### Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/amirabbas-gash/TAC-AI-Agent.git
   cd TAC-AI-Agent
   ```

2. **Install Dependencies**:

   ```bash
   pip install torch transformers sentence-transformers fitz pandas tqdm spacy numpy textwrap
   ```

3. **Install PyMuPDF**:

   ```bash
   pip install PyMuPDF
   ```

4. **Download Spacy Model**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Prepare Knowledge Base**:

   - Place your technical documentation PDF (e.g., `knowledge_base.pdf`) in the project directory.
   - Update the `pdf_path` variable in `rag_copy.py` to point to your PDF file.

6. **Hugging Face Authentication** (for Gemma-7B):

   - Ensure you have access to the `google/gemma-7b-it` model on Hugging Face.
   - Set up your Hugging Face token:

     ```bash
     export HF_TOKEN="your-hugging-face-token"
     ```

## Usage

1. **Run the Script**:

   ```bash
   python rag_copy.py
   ```

   This will:

   - Extract text from the PDF and preprocess it into sentence chunks.
   - Generate embeddings for the chunks and save them to `text_chunks_and_embeddings_df.csv`.
   - Load the Gemma-7B model and prepare it for inference.

2. **Query the Assistant**: Use the `answer_to_question` function to ask questions. Example:

   ```python
   response = answer_to_question("What is the firmware version for MES5448?")
   print(response)
   ```

## Example

**Query**: "What is the purpose of the MES5448 and MES7048 Data Center Switches?" **Response**:

```
The purpose of the MES5448 and MES7048 Data Center Switches is to provide high-performance networking solutions for data centers, supporting various interface types and advanced networking features suitable for aggregation and transport in carrier networks and data centers.
```

## Technical Details

- **PDF Processing**: Uses `PyMuPDF` to extract text and `spaCy` for sentence segmentation. Text is split into chunks of 10 sentences, with chunks under 30 tokens filtered out.
- **Embedding Model**: `all-mpnet-base-v2` generates 768-dimensional embeddings for semantic search, computed on CUDA for efficiency.
- **Retrieval**: Top-5 relevant chunks are retrieved using cosine similarity (dot product) between query and chunk embeddings.
- **LLM**: `google/gemma-7b-it` with 4-bit quantization (`BitsAndBytesConfig`) and Flash Attention (`flash_attention_2` or `sdpa`) for optimized inference.
- **Prompt Engineering**: Structured prompts include context from retrieved chunks and example-based instructions to guide the model.

## Challenges and Solutions

- **Challenge**: Irregular PDF formats (e.g., tables, multi-column text).
  - **Solution**: Custom `text_formatter` function to clean text and `spaCy` for accurate sentence splitting.
- **Challenge**: High memory usage for LLM inference.
  - **Solution**: 4-bit quantization and Flash Attention to reduce memory footprint.
- **Challenge**: Ensuring relevant retrieval for technical queries.
  - **Solution**: Fine-tuned chunk size (10 sentences) and minimum token filtering (30 tokens) to balance context and relevance.

## Future Improvements

- **Multi-lingual Support**: Add support for languages like Persian using multilingual embedding models (e.g., XLM-R) and language-specific preprocessing (e.g., Hazm).
- **Scalability**: Implement a vector database (e.g., FAISS or Pinecone) for faster retrieval with large knowledge bases.
- **Evaluation**: Add automated metrics (e.g., BLEU, ROUGE) to evaluate response quality.
- **RLHF Integration**: Incorporate Reinforcement Learning from Human Feedback to fine-tune response style based on engineer feedback.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bug fixes.


## Acknowledgments

- Built with Hugging Face Transformers, SentenceTransformers, and PyMuPDF.
- Inspired by the need to empower TAC engineers with fast, accurate access to technical documentation.
