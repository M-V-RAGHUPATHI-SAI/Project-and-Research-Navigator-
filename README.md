# 📚 Project & Research Navigator

**AI-Powered Knowledge Retrieval Engine for Academia**

---

## 🧠 Overview

**Project & Research Navigator** is an AI-powered **Retrieval-Augmented Generation (RAG)** system designed to help students, researchers, and educators quickly retrieve accurate information from academic documents such as **research papers (PDFs)** and **datasets (XLSX)**.

Instead of searching manually through large documents, users can **ask natural language questions** and receive:

* Context-aware answers
* Relevant document excerpts
* Similarity scores for transparency

This project was developed as part of **Tech Sprint** with a focus on **academic knowledge discovery**.

---

## 🚀 Key Features

* 🔍 **Semantic Search** using sentence embeddings
* 📄 **Multi-document support** (PDF & XLSX)
* 🧠 **RAG Pipeline** (Retriever + LLM)
* ⚡ **Fast similarity search** using ChromaDB
* 🤖 **LLM-powered answers** via Groq (LLaMA 3)
* 📊 **Source transparency** with similarity scores
* 🖥️ **Interactive UI** built using Streamlit

---

## 🏗️ System Architecture

```
User Query
   ↓
Sentence Embeddings (SentenceTransformer)
   ↓
Vector Store (ChromaDB)
   ↓
Top-K Relevant Chunks
   ↓
LLM (Groq – LLaMA 3)
   ↓
Final Answer + Retrieved Sources
```

---

## 🛠️ Tech Stack

| Layer            | Technology                    |
| ---------------- | ----------------------------- |
| Frontend         | Streamlit                     |
| Language Model   | Groq (LLaMA-3.1-8B)           |
| Embeddings       | SentenceTransformers (MiniLM) |
| Vector Database  | ChromaDB                      |
| Document Parsing | LangChain                     |
| Data Formats     | PDF, XLSX                     |
| Language         | Python                        |

---

## 📂 Project Structure

```
Project-and-Research-Navigator/
│── app.py                  # Streamlit frontend
│── ml_pipeline.py          # Core RAG pipeline
│── content/                # Academic documents (PDF/XLSX)
│── chroma_db/              # Persistent vector store
│── .env                    # Environment variables (not committed)
│── .gitignore
│── requirements.txt
│── README.md
```

---

## 📄 Supported Document Types

* ✅ PDF (`.pdf`)
* ✅ Excel (`.xlsx`)
* ❌ Images / Scanned PDFs (OCR not included)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/M-V-RAGHUPATHI-SAI/Project-and-Research-Navigator-.git
cd Project-and-Research-Navigator-
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure Environment Variables

Create a `.env` file in the root directory:

```env
API_KEY=your_groq_api_key_here
```

⚠️ **Never commit `.env` to GitHub**

---

### 5️⃣ Add Documents

Place your academic documents inside the `content/` folder:

```
content/
├── paper1.pdf
├── dataset.xlsx
```

---

### 6️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🧪 Example Questions

* *Summarize the key contributions of this research paper*
* *Explain the methodology used in this study*
* *What are the limitations of this research?*
* *What is a transformer architecture in deep learning?*
* *Explain qualitative vs quantitative research methods*

---

## 📊 Output Details

For every query, the system returns:

* ✅ AI-generated answer
* 📄 Retrieved document chunks
* 📈 Similarity scores
* 📌 Source file names

This ensures **transparency and trust** in responses.

---

## 🔐 Security & Best Practices

* API keys are stored using **environment variables**
* `.env`, `__pycache__`, and `.pyc` files are ignored via `.gitignore`
* GitHub Push Protection compliance ensured

---

## 🚧 Current Limitations

* No OCR for scanned PDFs
* Single-language support (English)
* No document upload via UI (folder-based ingestion)

---

## 🔮 Future Enhancements

* 📤 Upload documents directly from UI
* 📚 Citation formatting (APA / IEEE)
* 🧠 Domain-specific retrievers
* 📊 Confidence & relevance scoring
* 🌐 Multi-language support
* 🔁 Incremental document indexing

---

## 👥 Contributors

* **Venkata Raghupathi Sai Mannava**
* **A.Yaswant Sai**
* **K.Sarath Chandra**
* **V.Balaji Bhargav** 
* Team members – Tech Sprint

---

## 🏆 Use Cases

* Academic research assistance
* Literature review automation
* Student project analysis
* Institutional knowledge retrieval

---

## 📜 License

This project is developed for **educational and research purposes** under Tech Sprint.
License details can be added as required.

---

## ⭐ Acknowledgements

* LangChain
* Groq
* SentenceTransformers
* ChromaDB
* Streamlit
* Tech Sprint Organizers
