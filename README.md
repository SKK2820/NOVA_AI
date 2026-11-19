# NOVA - AI

### Your AI-Powered Research and Document Assistant

NOVA is a Python-based AI assistant powered by the **Google Gemini API**, built using Streamlit. It combines intelligent conversations, long-term memory, document understanding, semantic search, and multi-source web research in a single application.

NOVA is designed to go beyond ordinary AI conversations by allowing users to interact with their documents, recall previous discussions, conduct research across multiple web sources, and generate downloadable reports.

**Live Application:** Due to security reasons the deployment link will not be shared.

---

## Features

### 1. AI-Powered Conversations
- Intelligent conversational assistant powered by Google Gemini.
- Interactive chat interface.
- Start fresh conversations.
- Rename and manage previous conversations.
- Persistent chat history using SQLite in local deployments.

### 2. Long-Term Conversation Memory
- Rolling conversation summaries.
- Retains important information from previous discussions.
- Semantic and keyword-based retrieval of relevant past conversations.
- Recent conversation messages are retained for contextual continuity.
- User controls for managing conversation memory.

### 3. Document Intelligence
Upload and interact with your own documents.

Supported formats:
- PDF
- DOCX
- TXT

Capabilities:
- Extract text and tables from documents.
- Extract supported mathematical content from DOCX files.
- Split documents into searchable chunks.
- Build a local document index.
- Ask questions about uploaded documents.
- Retrieve relevant information using hybrid semantic and keyword search.
- Provide document-based citations in responses.

### 4. Advanced Research Mode
NOVA includes an advanced web research system designed to gather information from multiple online sources.

Features:
- Generates multiple search query variations.
- Searches across multiple web sources.
- Uses Google, Yahoo, and Bing as search-provider fallbacks.
- Removes duplicate sources.
- Encourages domain diversity.
- Extracts relevant webpage content when accessible.
- Falls back to available search snippets when full-page extraction is unsuccessful.
- Provides source references and research citations.
- Maintains research history.

### 5. Report Generation
Convert research and collected information into downloadable reports.

Supported formats:
- PDF
- DOCX
- Markdown

### 6. Usage Statistics
- Tracks input and output token usage.
- Tracks API request statistics.
- Records successful and failed requests.
- Maintains usage statistics in the local database.

### 7. Reliability and Backup
- Gemini API timeout and retry handling.
- TLS certificate verification using Certifi.
- Application logging.
- Automatic weekly SQLite database backups.
- Retains the latest eight automatic backups.
- Manual database backup and restore functionality.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Google Gemini API | AI model and conversation generation |
| SQLite | Local database and chat history |
| FastEmbed | Semantic embeddings |
| DDGS | Web search |
| PyPDF | PDF text extraction |
| pdfplumber | PDF text and table extraction |
| python-docx | DOCX processing |
| ReportLab | PDF report generation |
| Certifi | TLS certificate verification |

---

## Project Structure

```text
NOVA-AI/
│
├── app.py                 # Main Streamlit application
├── config.py              # API and model configuration
├── requirements.txt       # Python dependencies
├── NOVA_AI.jpg            # Application logo
├── README.md              # Project documentation
├── .gitignore             # Git ignored files
│
└── .streamlit/            # Optional Streamlit configuration
```

**Deployment Platform:** Streamlit Community Cloud

---

## Important Deployment Notes

NOVA uses SQLite and local filesystem storage for its database, documents, model cache, and backups.

Streamlit Community Cloud does not guarantee persistent local filesystem storage. Consequently, local chat history, uploaded documents, and database backups may not survive application restarts, rebuilds, or instance replacement.

For reliable long-term cloud storage, an external database and persistent file-storage solution would be required.

Keep API credentials in Streamlit Secrets or local environment variables. Never store secrets in source code or commit them to GitHub.

---

## Future Improvements

Potential areas for future development include:

- External database integration for persistent cloud storage.
- Persistent cloud document storage.
- Improved document retrieval and context handling.
- Enhanced research and source-verification capabilities.
- Additional export and productivity features.
- Further performance and reliability improvements.

---

## Author

**S Karthikeyan**

Created and developed as a personal AI assistant and research project.

**© 2026 S Karthikeyan. All Rights Reserved.**

---

### A Project Built to Explore the Possibilities of AI

**NOVA - AI | Think. Research. Discover.**
