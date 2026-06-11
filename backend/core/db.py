import sqlite3
from config import APP_DIR

DB_PATH = APP_DIR / "messages.db"

def init_db():
    APP_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                project_path TEXT NOT NULL,
                role TEXT NOT NULL,
                subtype TEXT,
                content TEXT NOT NULL,
                turn INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_project ON messages(project_path)")

def get_messages(project_path: str) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, role, subtype, content, turn FROM messages WHERE project_path = ? ORDER BY timestamp ASC",
            (project_path,)
        )
        return [dict(row) for row in cursor.fetchall()]

def save_message(project_path: str, msg: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO messages (id, project_path, role, subtype, content, turn)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(msg.get("id")),
                project_path,
                msg.get("role", "user"),
                msg.get("subtype"),
                msg.get("content", ""),
                msg.get("turn")
            )
        )

# Initialize the DB immediately when this module is imported
init_db()
