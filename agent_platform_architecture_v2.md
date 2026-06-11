# Technical Specification and Architecture: Desktop AI Multi-Agent Orchestration Platform

This document represents the complete architectural guide for developing an advanced desktop platform dedicated to the management and execution of complex tasks through AI agent networks. The system is designed to run locally, providing direct and secure access to the user's workspace files, optimizing workflows through dynamic model selection and proactive requirement clarification. THIS PROJECT WILL ONLY USE MODELS FROM OPENAI

---

## 1. General Concept and Architectural Objectives

The platform functions as an AI-assisted development environment, similar to the philosophy of modern intelligent workspace tools (Codex / Antigravity). The system's main objectives are:

* **Native and Direct Execution:** Eliminating heavy intermediate layers by running directly on the host operating system (e.g., Fedora Linux), allowing the AI to interact directly with the SSD and local tools (compilers, linters, testing tools).
* **Resource Optimization (Model Routing):** Automatically evaluating tasks to decide if a request can be solved by a small, ultra-fast, and cheap model or if it requires the reasoning of a top-tier model.
* **Determinism and Safety:** Structuring interactions through strict formats (JSON/XML) and running a closed verification cycle of the generated code before completing the task.

---

## 2. Conceptual Structure of the Monorepo

The project is organized as a simple monorepo, divided into two clear planes: the User Interface (Frontend) and the Intelligence Engine (Backend).

```
📁 agent-platform/
├── 📁 frontend/                  # Visual Layer and Desktop Wrapper
│   ├── package.json              # Configurations, dependencies, and start scripts
│   ├── main.js                   # Main Process (Native window management, OS dialogs)
│   └── 📁 src/                   # Web Graphical Interface (SPA)
│       ├── App.jsx               # Main UI state component
│       └── 📁 components/        # Chat, File Panel, Agent Graph (React Flow)
└── 📁 backend/                   # Control Layer and Agent Logic
    ├── main.py                   # Async server and connection management
    ├── requirements.txt          # Python dependencies (LLM SDKs, text processing)
    └── 📁 agents/                # Modules dedicated to the agent ecosystem
        ├── interrogator.py       # Proactive interrogation and refinement agent
        ├── router.py             # Prompt complexity classifier
        ├── supervisor.py         # Central hierarchical orchestrator (Manager)
        └── coder.py              # Source code writing and modification agent
```

---

## 3. Initial Project Indexing & Semantic RAG

When a new project folder is opened, the platform initiates a background indexing process. This phase is critical for providing the agents with a deep understanding of the codebase structure and logic without overwhelming the model's context window.

### A. Intelligent Traversal & File Tree Generation
The first step is a selective scan of the workspace. To avoid indexing "noise," the system implements a smart filtering mechanism:
*   **Git-Aware Filtering:** The backend parses the project's `.gitignore` file to skip dependency folders (e.g., `node_modules`, `venv`), binary files, and hidden system directories.
*   **Tree Representation:** A JSON map of the filtered file structure (Project File Tree) is generated and cached. This map serves as the primary structural context for the Interrogator Agent.

### B. Language-Aware Semantic Chunking (AST)
Instead of arbitrary text splitting, the platform uses a semantic approach to break down source code into manageable "chunks":
*   **Abstract Syntax Tree (AST) Splitting:** Using language-specific parsers, the system identifies logical boundaries such as functions, classes, and methods.
*   **Context Preservation:** Each chunk represents a complete and isolated logical block. This ensures that when a piece of code is retrieved, it retains its functional meaning, preventing the loss of logic that occurs with character-based splitting.

### C. Local Vector Database (ChromaDB)
To enable fast and efficient retrieval, the system builds a local, serverless vector database:
*   **Embedded Persistence:** The platform utilizes **ChromaDB** in persistent mode, running directly within the Python process and saving data to a local SQLite-backed file.
*   **OpenAI Embeddings:** Code chunks are transformed into vectors using the `text-embedding-3-small` model. These vectors are then stored in ChromaDB for similarity-based retrieval.

### D. Asynchronous Indexing Workflow
The indexing process is designed to be non-blocking to maintain a fluid user experience:
1.  **Trigger:** The user selects a project folder via the UI.
2.  **Background Processing:** FastAPI initiates an asynchronous task for indexing.
3.  **Real-Time Feedback:** The backend streams progress updates (e.g., "Indexed 45/120 files") to the Electron frontend via **WebSockets**.
4.  **Ready State:** Once complete, the vector database is available for the Supervisor and Interrogator agents to perform high-precision RAG (Retrieval-Augmented Generation).

---

## 4. "Interrogate Me" Phase (Requirement Clarification)

Inspired by grill me feature from claude code, this is the first stage of any interaction. Instead of being a passive assistant, the system acts as a proactive partner. When a user provides a prompt, the system does not start execution directly but initiates an interrogation session to ensure all technical details are clear.

### Hybrid Context Strategy (How the refinement model "thinks"):
To ask extremely targeted and intelligent questions without generating latency, a lightweight model (GPT-5.4 mini) is provided with a context structured on three levels:

1.  **Project File Tree:** A complete map of directories and files in the current workspace. This provides the model with a structural understanding of the technologies used.
2.  **Active Files State (Git Context):** Identifying currently open files in the editor or recently modified ones by querying the local repository state. These files are read in full and attached to the context.
3.  **Local Dynamic RAG:** A lightweight local vector database is used only to extract isolated function definitions or table schemas relevant to the keywords in the prompt. (the vector db that starts up when a project is opened)

### Stopping Criteria:
The interrogator returns responses in a strictly structured format (valid JSON). This contains a boolean indicator (`is_prompt_ready`). As long as this indicator is false, the interface displays the generated question and waits for the user's response. When the AI considers it has all the necessary technical details (target files, desired behavior, dependencies), the indicator becomes true, and the final prompt, enriched and refined, is sent to the routing layer.

---

## 5. Automated Routing Layer & Agent Selection

Once the prompt is finalized through the "Interrogate Me" phase, the platform determines the most efficient way to execute the task. This layer manages both **Model Routing** (cost/latency) and **Agent Delegation** (specialization) using **LangChain** and **OpenAI**.

### Semantic Routing (LangChain Integrated):
1.  **Vector Conversion:** The refined prompt is converted into a high-dimensional vector using OpenAI's `text-embedding-3-small` model.
2.  **Complexity & Intent Classification:** The system utilizes a LangChain-based **Semantic Router**. The prompt vector is compared against curated "route descriptors" to categorize the task.
3.  **Routing Decision (OpenAI Ecosystem):**
    *   **Simple/Generic Tasks:** Directed back to `GPT-4o-mini` for immediate response. This minimizes costs and provides near-instant responses for conversational queries.
    *   **Specialized Technical Tasks:** Delegated to the **Supervisor Agent**, which orchestrates specialized agents (like the Coder Agent) using high-reasoning models (`GPT-4o` or `o1`).

---

## 6. Coding Agent Architecture: "Plan -> Code -> Verify" Cycle

The execution phase is handled by a high-reasoning model (Coding Agent) that operates within a structured environment. To ensure production-grade stability and code quality, the system enforces a rigorous multi-step lifecycle.

### A. Context Orchestration: The XML Knowledge Tree
Instead of sending raw, unorganized text, the platform assembles a dynamic **Knowledge Tree** for the agent. This structure uses XML tags to provide clear boundaries and high semantic density:
*   **System Instructions (The Rules):** Defines the project's coding standards (e.g., "Use async FastAPI patterns," "Strict type-hinting," "Repository pattern implementation").
*   **The Repo Map (Skeleton):** A high-level overview of the codebase using class and function signatures (extracted via AST parsers). This allows the model to understand global dependencies without reading every file in full.
*   **Target Files:** The specific files marked for modification or creation, wrapped in `<file path="...">` tags.
*   **Specific Context (RAG):** Relevant internal documentation, DDL schemas (critical for database tasks), or connected API routes retrieved from the vector database.

### B. The Three-Phase Execution Workflow
The agent is forced to follow a **Chain-of-Thought (CoT)** pattern, separating architectural thinking from implementation.

1.  **Phase 1: Planning (Attack Plan):**
    The agent must first generate a **Plan of Attack** in Markdown. It describes which files will be modified, what logic will be added, and how it affects the system. This allows for a checkpoint where execution can be halted if the logic is flawed.
2.  **Phase 2: Implementation via Unified Diffs:**
    To optimize speed and prevent code loss, the agent does not rewrite entire files. Instead, it uses a **Search & Replace XML block** format (similar to Aider/Cursor).
    *   *Example:* `<change><search>old_code</search><replace>new_code</replace></change>`
    The Python backend parses these blocks and applies modifications programmatically.
3.  **Phase 3: Automated Validation (The Sandbox Loop):**
    The system enters a background verification cycle:
    *   **Linter/Syntax Check:** Runs static analysis tools (e.g., `flake8`, `mypy`, `ruff`) over modified files.
    *   **Automated Testing:** Launches the local test suite (e.g., `pytest`) in an isolated environment.
    *   **Self-Correction:** If errors occur (non-zero exit codes), the `stderr` is captured and sent back to the agent: *"Your code produced this error: [Error]. Please correct it."* The agent has up to 3 autonomous attempts to fix the code before finalizing.

### C. Agent Toolset (Function Calling)
The Coding Agent is equipped with a specialized set of tools to interact with the workspace:
*   `read_file(path, line_start, line_end)`: Surgical reading of specific code segments.
*   `write_file(path, content)`: Creating new modules or documentation.
*   `patch_file(path, search_block, replace_block)`: Applying targeted modifications to existing files.
*   `search_grep(pattern)`: Global workspace search for variable definitions or function usages.

---

## 7. Local Running and Connection (Control Plane vs. Data Plane)

The application runs entirely on the user's laptop, ensuring an asynchronous, fluid experience.

### Control Plane (FastAPI Backend Process)
The Python backend functions as a central orchestrator, exposing HTTP endpoints and a permanent connection through **WebSockets** for real-time state updates (e.g., *"Interrogating user"*, *"Evaluating complexity"*, *"Coding agent is writing the plan"*).

### Data Plane (Workspace and Native Execution)
Execution and modifications take place directly in the physical folder selected by the user, using asynchronous system processes to invoke local commands.

---

## 8. Advanced Extensions for Security and Observability

1.  **"Security Guardrail" Agent (DevSecOps):** Scans code blocks generated by the AI before they are written to disk to intercept vulnerabilities.
2.  **Agent Graph Visualization (Observability):** The Electron web interface uses a dynamic graph library to render the decision flow in real-time. The user sees how the task moves from the **Interrogator Agent** to the **Supervisor** or **Coder Agent**.
3.  **Time-Travel Rollback (Safety System):** Creates a quick local snapshot or Git branch before modifications, allowing a complete "Undo" from a single click.

---

## 9. Secure Credential Management (API Key Storage)

To ensure maximum security for sensitive information such as OpenAI API keys, the platform avoids insecure storage methods like local text files or browser storage. Instead, it leverages the host operating system's native security infrastructure.

*   **OS-Level Integration:** The backend utilizes the system's native credential manager to store and retrieve sensitive keys. This ensures that credentials are encrypted at rest and protected by the operating system's security policies.
*   **Platform-Specific Backends:**
    *   **macOS:** Integration with the **Apple Keychain**.
    *   **Windows:** Integration with the **Windows Credential Locker**.
    *   **Linux (Fedora/GNOME):** Integration with the **Secret Service API** (Libsecret).
*   **Workflow:** When a user provides an API key through the interface, it is transmitted over a secure local channel to the backend, which then delegates the storage responsibility to the OS. The key is retrieved only during active execution cycles when needed for model authentication, ensuring it never persists in plain text within the application folder.
