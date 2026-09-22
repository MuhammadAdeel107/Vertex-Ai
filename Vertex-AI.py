"""
Vertex AI - Ollama Chatbot (Gradio version)
Sidebar: status, new chat, model picker, temperature, system prompt,
searchable chat history, delete / clear all.
Tabs: Dashboard (metrics + charts) and Chat (streaming, regenerate, export).
"""

import os
import uuid
import sqlite3
from datetime import datetime, timedelta
from contextlib import contextmanager

import gradio as gr
import ollama
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DB_FILE = "chatbot.db"
DEFAULT_MODEL = "gemma3:1b"
RECENT_CONTEXT_MESSAGES = 6
NUM_CTX = 2048
NUM_PREDICT = 256


# ============================================================
# DATABASE
# ============================================================

@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                model TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at)"
        )
        conn.commit()


init_database()


def create_conversation(model):
    conversation_id = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (id, title, model, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, "New Conversation", model, now, now),
        )
        conn.commit()
    return conversation_id


def get_conversations(search_term=None):
    with get_connection() as conn:
        cursor = conn.cursor()
        if search_term:
            cursor.execute(
                "SELECT id, title, model, created_at, updated_at FROM conversations "
                "WHERE title LIKE ? ORDER BY updated_at DESC",
                (f"%{search_term}%",),
            )
        else:
            cursor.execute(
                "SELECT id, title, model, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC"
            )
        return cursor.fetchall()


def get_messages(conversation_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        )
        rows = cursor.fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def save_message(conversation_id, role, content):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        cursor.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
        )
        conn.commit()


def update_conversation_title(conversation_id, title):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id)
        )
        conn.commit()


def delete_conversation(conversation_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()


def delete_all_conversations():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM conversations")
        conn.commit()


def delete_last_exchange(conversation_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM messages WHERE id IN (
                SELECT id FROM messages WHERE conversation_id = ?
                ORDER BY id DESC LIMIT 2
            )
            """,
            (conversation_id,),
        )
        conn.commit()


def count_messages():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages")
        return cursor.fetchone()[0]


def get_messages_dataframe():
    with get_connection() as conn:
        try:
            df = pd.read_sql_query(
                """
                SELECT m.id, m.role, m.content, m.created_at, c.model, c.title
                FROM messages m JOIN conversations c ON m.conversation_id = c.id
                """,
                conn,
            )
        except Exception:
            df = pd.DataFrame(columns=["id", "role", "content", "created_at", "model", "title"])
    return df


# ============================================================
# OLLAMA
# ============================================================

def get_models():
    try:
        result = ollama.list()
        return [m for m in result.models if getattr(m, "model", None)]
    except Exception:
        return []


def get_model_name(model):
    return getattr(model, "model", None) or "Unknown"


def get_model_size_bytes(model):
    return getattr(model, "size", 0) or 0


def format_size(size_bytes):
    if not size_bytes:
        return "Unknown"
    size_gb = size_bytes / (1024 ** 3)
    if size_gb >= 1:
        return f"{size_gb:.1f} GB"
    return f"{size_bytes / (1024 ** 2):.0f} MB"


def check_ollama_running():
    try:
        ollama.list()
        return True
    except Exception:
        return False


# ============================================================
# HELPERS THAT BRIDGE DB <-> UI
# ============================================================

def conversation_choices(search_term=None):
    """Return [(label, id), ...] for the Radio list, newest first."""
    rows = get_conversations(search_term=search_term or None)
    choices = []
    for conversation_id, title, *_ in rows:
        label = title if len(title) <= 40 else title[:40] + "..."
        choices.append((label, conversation_id))
    return choices


def get_title(conversation_id):
    for row in get_conversations():
        if row[0] == conversation_id:
            return row[1]
    return "New Conversation"


def history_to_chatbot(conversation_id):
    return get_messages(conversation_id)


def status_text():
    return "🟢 Ollama Online" if check_ollama_running() else "🔴 Ollama Offline"


# ============================================================
# EVENT HANDLERS
# ============================================================

def on_load():
    online = check_ollama_running()
    models = get_models() if online else []
    model_names = [get_model_name(m) for m in models]
    default_model = model_names[0] if model_names else DEFAULT_MODEL

    rows = get_conversations()
    if rows:
        conv_id = rows[0][0]
    else:
        conv_id = create_conversation(default_model)

    choices = conversation_choices()

    return (
        conv_id,
        status_text(),
        gr.update(choices=model_names, value=default_model if model_names else None),
        gr.update(choices=choices, value=conv_id),
        history_to_chatbot(conv_id),
        f"### 💬 {get_title(conv_id)}",
    )


def new_chat(model):
    conv_id = create_conversation(model or DEFAULT_MODEL)
    choices = conversation_choices()
    return (
        conv_id,
        gr.update(choices=choices, value=conv_id),
        [],
        f"### 💬 {get_title(conv_id)}",
    )


def select_chat(conversation_id):
    if not conversation_id:
        return [], "### 💬 New Conversation"
    return history_to_chatbot(conversation_id), f"### 💬 {get_title(conversation_id)}"


def search_chats(search_term, current_id):
    choices = conversation_choices(search_term)
    valid_ids = [c[1] for c in choices]
    value = current_id if current_id in valid_ids else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value)


def delete_selected(conversation_id, model):
    if conversation_id:
        delete_conversation(conversation_id)

    rows = get_conversations()
    if rows:
        new_id = rows[0][0]
    else:
        new_id = create_conversation(model or DEFAULT_MODEL)

    choices = conversation_choices()
    return (
        new_id,
        gr.update(choices=choices, value=new_id),
        history_to_chatbot(new_id),
        f"### 💬 {get_title(new_id)}",
    )


def clear_all(model):
    delete_all_conversations()
    new_id = create_conversation(model or DEFAULT_MODEL)
    choices = conversation_choices()
    return (
        new_id,
        gr.update(choices=choices, value=new_id),
        [],
        f"### 💬 {get_title(new_id)}",
    )


def export_chat(conversation_id):
    messages = get_messages(conversation_id)
    title = get_title(conversation_id)
    text = "\n\n".join(f"**{m['role'].capitalize()}:** {m['content']}" for m in messages)

    safe_name = (title[:30].strip() or "chat").replace("/", "-").replace("\\", "-")
    path = f"/tmp/{safe_name}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def respond(message, chat_history, conversation_id, model, temperature, system_prompt):
    if not message or not message.strip():
        yield chat_history, "", f"### 💬 {get_title(conversation_id)}"
        return

    if not model:
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ No model selected/available. Check the sidebar."},
        ]
        yield chat_history, "", f"### 💬 {get_title(conversation_id)}"
        return

    save_message(conversation_id, "user", message)

    if get_title(conversation_id) == "New Conversation":
        title = message.strip()
        title = title[:35] + "..." if len(title) > 35 else title
        update_conversation_title(conversation_id, title)

    chat_history = chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    yield chat_history, "", f"### 💬 {get_title(conversation_id)}"

    recent = get_messages(conversation_id)[-RECENT_CONTEXT_MESSAGES:]
    ollama_messages = [{"role": "system", "content": system_prompt}]
    ollama_messages += [{"role": m["role"], "content": m["content"]} for m in recent]

    full_response = ""
    try:
        stream = ollama.chat(
            model=model,
            messages=ollama_messages,
            stream=True,
            options={
                "temperature": temperature,
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
            },
        )
        for chunk in stream:
            full_response += chunk["message"]["content"]
            chat_history[-1]["content"] = full_response
            yield chat_history, "", f"### 💬 {get_title(conversation_id)}"

    except ollama.ResponseError as error:
        full_response = f"❌ Model error: {error}"
        chat_history[-1]["content"] = full_response
        yield chat_history, "", f"### 💬 {get_title(conversation_id)}"
    except Exception as error:
        full_response = f"❌ Could not generate a response.\n\n```\n{error}\n```"
        chat_history[-1]["content"] = full_response
        yield chat_history, "", f"### 💬 {get_title(conversation_id)}"

    if full_response.strip():
        save_message(conversation_id, "assistant", full_response)

    yield chat_history, "", f"### 💬 {get_title(conversation_id)}"


def regenerate(chat_history, conversation_id, model, temperature, system_prompt):
    if not chat_history or chat_history[-1]["role"] != "assistant":
        yield chat_history, f"### 💬 {get_title(conversation_id)}"
        return

    delete_last_exchange(conversation_id)
    last_user_msg = chat_history[-2]["content"] if len(chat_history) >= 2 else ""
    chat_history = chat_history[:-2]

    for updated_history, _, title in respond(
        last_user_msg, chat_history, conversation_id, model, temperature, system_prompt
    ):
        yield updated_history, title


def refresh_dashboard():
    conversations_all = get_conversations()
    conversation_count = len(conversations_all)
    message_count = count_messages()
    df = get_messages_dataframe()

    online = check_ollama_running()
    models = get_models() if online else []
    model_names = [get_model_name(m) for m in models]

    avg_len = int(df["content"].str.len().mean()) if not df.empty else 0

    models_df = pd.DataFrame(
        {
            "Model": model_names,
            "Size": [format_size(get_model_size_bytes(m)) for m in models],
        }
    ) if models else pd.DataFrame(columns=["Model", "Size"])

    if df.empty:
        empty = pd.DataFrame({"x": [], "y": []})
        return (
            len(model_names),
            conversation_count,
            message_count,
            f"{avg_len} chars",
            models_df,
            empty,
            empty,
            empty,
            empty,
            pd.DataFrame(columns=["Time", "Role", "Conversation", "Model"]),
        )

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["date"] = df["created_at"].dt.date

    last_14_days = [(datetime.now() - timedelta(days=i)).date() for i in range(13, -1, -1)]
    daily_counts = (
        df[df["date"].isin(last_14_days)]
        .groupby("date")
        .size()
        .reindex(last_14_days, fill_value=0)
    )
    daily_df = pd.DataFrame(
        {"Day": [d.strftime("%m-%d") for d in daily_counts.index], "Messages": daily_counts.values}
    )

    role_counts = df["role"].value_counts()
    role_df = pd.DataFrame({"Role": role_counts.index, "Messages": role_counts.values})

    model_counts = (
        df[df["role"] == "user"]["model"].value_counts() if "model" in df.columns else pd.Series(dtype=int)
    )
    model_df = pd.DataFrame({"Model": model_counts.index, "Messages": model_counts.values})

    longest = df.groupby("title").size().sort_values(ascending=False).head(5)
    longest_df = pd.DataFrame({"Conversation": longest.index, "Messages": longest.values})

    recent = (
        df.sort_values("created_at", ascending=False)
        .head(10)[["created_at", "role", "title", "model"]]
        .rename(columns={"created_at": "Time", "role": "Role", "title": "Conversation", "model": "Model"})
    )

    return (
        len(model_names),
        conversation_count,
        message_count,
        f"{avg_len} chars",
        models_df,
        daily_df,
        role_df,
        model_df,
        longest_df,
        recent,
    )


# ============================================================
# CSS
# ============================================================

CSS = """
.status-line { font-weight: 600; margin-bottom: 4px; }
.sidebar-col { border-right: 1px solid var(--border-color-primary); padding-right: 16px; }
"""

THEME_JS = """
() => {
    const container = document.querySelector('.gradio-container');
    if (container) {
        container.classList.toggle('dark');
    }
}
"""


# ============================================================
# UI
# ============================================================

with gr.Blocks(title="Vertex AI") as demo:

    conversation_id_state = gr.State(None)

    with gr.Row():
        gr.Markdown("# 🏠 Vertex AI\nRun AI locally with Ollama")
        theme_toggle_btn = gr.Button("🌗 Light / Dark", scale=0, size="sm")

    with gr.Row():

        # ---------------- SIDEBAR ----------------
        with gr.Column(scale=1, min_width=280, elem_classes="sidebar-col"):
            status_box = gr.Markdown("🟢 Ollama Online", elem_classes="status-line")

            new_chat_btn = gr.Button("➕ New Chat", variant="primary")

            gr.Markdown("### 🧠 AI Model")
            model_dropdown = gr.Dropdown(choices=[], label="Select Model")

            gr.Markdown("### ⚙️ Settings")
            temperature_slider = gr.Slider(0.0, 1.5, value=0.5, step=0.1, label="Temperature")
            system_prompt_box = gr.Textbox(
                label="System Prompt",
                value=(
                    "You are a helpful AI assistant. "
                    "Give clear, accurate and useful answers. "
                    "Use Markdown when appropriate."
                ),
                lines=5,
            )

            gr.Markdown("### 💬 Chat History")
            search_box = gr.Textbox(placeholder="🔍 Search by title...", show_label=False)
            chat_list = gr.Radio(choices=[], label="Conversations")

            with gr.Row():
                delete_btn = gr.Button("🗑️ Delete selected", size="sm")
                clear_all_btn = gr.Button("🧨 Clear all", size="sm")

            gr.Markdown("💾 Chats are saved locally in `chatbot.db`")

        # ---------------- MAIN ----------------
        with gr.Column(scale=4):

            with gr.Tabs():

                # ---------------- DASHBOARD ----------------
                with gr.Tab("📊 Dashboard"):

                    refresh_btn = gr.Button("🔄 Refresh dashboard")

                    with gr.Row():
                        metric_models = gr.Number(label="🤖 Installed Models", interactive=False)
                        metric_conversations = gr.Number(label="💬 Conversations", interactive=False)
                        metric_messages = gr.Number(label="📝 Total Messages", interactive=False)
                        metric_avg_len = gr.Textbox(label="✍️ Avg. Message Length", interactive=False)

                    gr.Markdown("### 🤖 Your Ollama Models")
                    models_table = gr.Dataframe(headers=["Model", "Size"], interactive=False)

                    gr.Markdown("### 📈 Usage Analytics")
                    with gr.Row():
                        chart_daily = gr.BarPlot(x="Day", y="Messages", title="Messages over the last 14 days")
                        chart_role = gr.BarPlot(x="Role", y="Messages", title="Messages by role")
                    with gr.Row():
                        chart_model = gr.BarPlot(x="Model", y="Messages", title="Usage by model")
                        chart_longest = gr.BarPlot(x="Conversation", y="Messages", title="Longest conversations")

                    gr.Markdown("**Recent activity**")
                    recent_table = gr.Dataframe(
                        headers=["Time", "Role", "Conversation", "Model"], interactive=False
                    )

                # ---------------- CHAT ----------------
                with gr.Tab("💬 Chat"):

                    with gr.Row():
                        chat_title = gr.Markdown("### 💬 New Conversation")
                        export_btn = gr.Button("⬇️ Export", size="sm")
                    export_file = gr.File(label="Exported chat", visible=False)

                    chatbot = gr.Chatbot(height=320)

                    with gr.Row():
                        message_box = gr.Textbox(
                            placeholder="Ask your local AI anything...",
                            show_label=False,
                            scale=5,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    regenerate_btn = gr.Button("🔄 Regenerate last response")

    # ============================================================
    # WIRING
    # ============================================================

    dashboard_outputs = [
        metric_models, metric_conversations, metric_messages, metric_avg_len,
        models_table, chart_daily, chart_role, chart_model, chart_longest, recent_table,
    ]

    demo.load(
        fn=on_load,
        outputs=[conversation_id_state, status_box, model_dropdown, chat_list, chatbot, chat_title],
    ).then(fn=refresh_dashboard, outputs=dashboard_outputs)

    new_chat_btn.click(
        fn=new_chat,
        inputs=[model_dropdown],
        outputs=[conversation_id_state, chat_list, chatbot, chat_title],
    )

    chat_list.change(
        fn=select_chat,
        inputs=[chat_list],
        outputs=[chatbot, chat_title],
    )
    chat_list.change(fn=lambda x: x, inputs=[chat_list], outputs=[conversation_id_state])

    search_box.change(
        fn=search_chats,
        inputs=[search_box, conversation_id_state],
        outputs=[chat_list],
    )

    delete_btn.click(
        fn=delete_selected,
        inputs=[conversation_id_state, model_dropdown],
        outputs=[conversation_id_state, chat_list, chatbot, chat_title],
    )

    clear_all_btn.click(
        fn=clear_all,
        inputs=[model_dropdown],
        outputs=[conversation_id_state, chat_list, chatbot, chat_title],
    )

    export_btn.click(fn=export_chat, inputs=[conversation_id_state], outputs=[export_file]).then(
        fn=lambda: gr.update(visible=True), outputs=[export_file]
    )

    send_btn.click(
        fn=respond,
        inputs=[message_box, chatbot, conversation_id_state, model_dropdown, temperature_slider, system_prompt_box],
        outputs=[chatbot, message_box, chat_title],
    ).then(fn=refresh_dashboard, outputs=dashboard_outputs)

    message_box.submit(
        fn=respond,
        inputs=[message_box, chatbot, conversation_id_state, model_dropdown, temperature_slider, system_prompt_box],
        outputs=[chatbot, message_box, chat_title],
    ).then(fn=refresh_dashboard, outputs=dashboard_outputs)

    regenerate_btn.click(
        fn=regenerate,
        inputs=[chatbot, conversation_id_state, model_dropdown, temperature_slider, system_prompt_box],
        outputs=[chatbot, chat_title],
    ).then(fn=refresh_dashboard, outputs=dashboard_outputs)

    refresh_btn.click(fn=refresh_dashboard, outputs=dashboard_outputs)

    theme_toggle_btn.click(fn=None, js=THEME_JS)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=CSS)