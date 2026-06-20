# InterroAI

InterroAI is a local desktop AI multi-agent orchestration platform. It is designed as an AI-assisted development environment that runs locally, giving the AI direct and secure access to your workspace files to optimize workflows through dynamic model selection and proactive requirement clarification.

## Architecture & Stack

- **Frontend:** React 18, Vite, and Electron. It provides native window management and an interactive chat interface.
- **Backend:** Python 3.11+ with FastAPI. It uses asynchronous HTTP endpoints and WebSockets to drive the orchestration of the agents.

## Features

1. **Initial Project Indexing & Semantic RAG**
   - Two-phase asynchronous project indexing: 
     - *Phase 1:* Fast file-tree generation respecting `.gitignore` files.
     - *Phase 2:* AST and language-aware chunking. The chunks are embedded using OpenAI's `text-embedding-3-small` and stored in a persistent local ChromaDB instance (`~/.interroai/chroma/`).
2. **"Interrogate Me" Phase (Grill Agent)**
   - Before writing code, the system acts as a proactive partner to clarify technical requirements. It utilizes a hybrid context (file tree + git context + ChromaDB RAG) to ask clarifying questions before execution. 
3. **Automated Routing Layer & Intent Classification**
   - Inputs are first classified by intent (`answer`, `interrogate`, `implement`).
   - If auto-routing is enabled, a local offline classifier (TF-IDF + Logistic Regression) categorizes the prompt's complexity to select the most appropriate AI model tier (ranging from lightweight to high-effort reasoning models). This provides sub-millisecond, offline routing.
4. **Coding Agent: "Plan -> Code -> Verify" Cycle**
   - *Phase 1 (Plan):* The agent formulates an Attack Plan describing the required modifications.
   - *Phase 2 (Code):* Issues XML-based search/replace blocks applied programmatically via a unified diff patcher.
   - *Phase 3 (Verify):* A local sandbox invokes static analysis (`ruff`) and local test suites (`pytest`), feeding errors back to the model for autonomous self-correction (up to 3 attempts).
5. **Secure Credential Management**
   - To ensure maximum security, OpenAI API keys are not stored in plaintext config files. InterroAI uses the OS-native keychain (via Python's `keyring` module), ensuring keys are encrypted at rest (macOS Keychain, Windows Credential Locker, etc.).

## Local Development & Setup

Make sure you have Node.js and Python 3.11+ installed.

1. **Install Backend Dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Install Frontend Dependencies:**
   ```bash
   cd frontend
   npm install
   ```

3. **Run the Full Stack:**
   From the `frontend/` directory, run:
   ```bash
   npm run dev
   ```
   This will simultaneously start the FastAPI backend, the Vite development server, and launch the Electron application.
