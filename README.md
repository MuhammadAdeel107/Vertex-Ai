# Vertex AI - Local Ollama Chatbot

A feature-rich, local AI chat application developed with Python, Gradio, and Ollama. This application provides a modern sidebar interface, persistent SQLite conversation history, real-time streaming responses, and an integrated analytics dashboard.

---

## Features

- **Local LLM Integration:** Execute models locally using Ollama (compatible with models such as Gemma, Qwen, and Llama).
- **Persistent Storage:** All conversation threads and messages are securely stored locally via SQLite (`chatbot.db`).
- **Searchable History:** Filter and locate past chat sessions instantly using the sidebar search functionality.
- **Analytics Dashboard:** Monitor usage metrics, total messages, active models, and view visual analytics charts for daily activity, role distribution, and conversation length.
- **Configurable Parameters:** Adjust generation temperature and system prompts dynamically.
- **Theme Customization:** Integrated light and dark mode toggles.
- **Export Capability:** Export any active conversation directly to a Markdown file.

---

## Tech Stack

- **Frontend & UI:** Gradio
- **AI Backend:** Ollama Python SDK
- **Data Processing:** Pandas
- **Database:** SQLite3

---

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/MuhammadAdeel107/Vertex-Ai.git](https://github.com/MuhammadAdeel107/Vertex-Ai.git)
   cd Vertex-Ai
   Verify Ollama Status:
Ensure Ollama is installed and running on your system, and pull your target model:

Bash
ollama pull gemma3:1b
Install Dependencies:
Use uv (recommended) or pip to install the required packages:

Bash
# Using uv
uv sync

# Using pip
pip install -r requirements.txt
Running the Application
Execute the main script to launch the local web interface:

Bash
python Vertex-AI.py
Open your browser and navigate to http://127.0.0.1:7860.

Project Structure
Plaintext
muhammadadeel107-vertex-ai/
├── Vertex-AI.py        # Main application script
├── chatbot.db          # Local SQLite database (auto-generated)
├── pyproject.toml      # Project configuration and dependencies
├── requirements.txt    # Pip dependencies list
├── README.md           # Project documentation
└── src/
    └── vertex_ai/
        └── __init__.py # Package initialization
