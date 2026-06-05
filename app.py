import json
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from llm import ask_llm
from pdf_utils import extract_text_from_pdf, split_text
from vector_db import build_vector_db, search, load_vector_db, has_knowledge_base, clear_vector_db

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
MEMORY_FILE = DATA_DIR / "chat_memory.json"
STATE_FILE = DATA_DIR / "app_state.json"

UPLOAD_FOLDER.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}

try:
    load_vector_db()
except Exception as e:
    print(f"[WARN] Failed to load vector DB: {e}")


PERSONALITY_PROMPTS = {
    "professional": {
        "zh": "你是专业、清晰、可靠的知识库助手。回答要准确、有条理，适合校招项目展示。",
        "en": "You are a professional, clear, and reliable knowledge base assistant. Be accurate, structured, and suitable for a portfolio project.",
    },
    "mentor": {
        "zh": "你是温柔、耐心、鼓励式的学习导师。解释要照顾初学者，多用简单例子。",
        "en": "You are a warm, patient, encouraging mentor. Explain for beginners and use simple examples.",
    },
    "concise": {
        "zh": "你是高效极简助手。只说重点，少废话，用最短的话给出可执行答案。",
        "en": "You are a concise and efficient assistant. Focus on key points and give actionable answers with minimal wording.",
    },
    "academic": {
        "zh": "你是学术严谨型助手。回答要严谨、客观，适合论文、报告和课程资料总结。",
        "en": "You are an academically rigorous assistant. Be objective, precise, and suitable for papers, reports, and course materials.",
    },
    "pm": {
        "zh": "你像产品经理一样回答。重视目标、用户场景、方案拆解、优先级和落地步骤。",
        "en": "You answer like a product manager. Focus on goals, user scenarios, solution breakdown, priorities, and implementation steps.",
    },
    "coding": {
        "zh": "你是编程老师。解释代码、项目结构和报错时要一步一步讲清楚。",
        "en": "You are a programming teacher. Explain code, project structure, and errors step by step.",
    },
}

ANSWER_STYLE_PROMPTS = {
    "standard": {
        "zh": "回答长度适中，结构清楚。",
        "en": "Use a moderate length and a clear structure.",
    },
    "short": {
        "zh": "回答尽量简短，只保留核心信息。",
        "en": "Keep the answer short and include only the essential information.",
    },
    "detailed": {
        "zh": "回答要详细，必要时分步骤解释。",
        "en": "Give a detailed answer and explain step by step when useful.",
    },
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_memory() -> List[Dict[str, str]]:
    data = read_json(MEMORY_FILE, [])
    return data if isinstance(data, list) else []


def save_memory(memory: List[Dict[str, str]]) -> None:
    # Limit memory size so prompts do not become too long.
    write_json(MEMORY_FILE, memory[-60:])


def load_state() -> Dict[str, Any]:
    data = read_json(STATE_FILE, {})
    return data if isinstance(data, dict) else {}


def save_state(state: Dict[str, Any]) -> None:
    write_json(STATE_FILE, state)


def normalize_language(language: str) -> str:
    return "en" if language == "en" else "zh"


def build_messages(
    question: str,
    context_chunks: List[Dict[str, Any]],
    personality: str,
    answer_style: str,
    mode: str,
    knowledge_name: str,
    language: str,
    memory: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    language = normalize_language(language)
    personality_key = personality if personality in PERSONALITY_PROMPTS else "professional"
    style_key = answer_style if answer_style in ANSWER_STYLE_PROMPTS else "standard"

    personality_text = PERSONALITY_PROMPTS[personality_key][language]
    style_text = ANSWER_STYLE_PROMPTS[style_key][language]

    if language == "en":
        lang_rule = "You must reply entirely in English. Do not use Chinese in your answer, labels, headings, or explanations."
        no_context = "No relevant content was retrieved from the knowledge base."
        kb_name = knowledge_name or "AI Knowledge Base"
        task = (
            "You are an AI knowledge base assistant. Prioritize the retrieved knowledge base content. "
            "If the content does not contain the answer, clearly say that no relevant content was found in the knowledge base. "
            "Do not invent sources or pretend that the document contains something it does not."
        )
        direct_task = "You are an AI assistant. Answer the user's question directly."
        history_title = "Conversation memory"
        context_title = "Retrieved knowledge base content"
        question_title = "User question"
    else:
        lang_rule = "你必须全程使用中文回答，不要在标题、选项说明或正文中夹杂英文，除非用户要求解释英文术语。"
        no_context = "没有从知识库中检索到相关内容。"
        kb_name = knowledge_name or "AI 知识库"
        task = (
            "你是一个 AI 知识库检索助手。请优先基于知识库检索内容回答。"
            "如果知识库内容中没有相关信息，请明确说明“知识库中没有找到相关内容”，不要编造资料来源。"
        )
        direct_task = "你是一个 AI 助手，请直接回答用户问题。"
        history_title = "长期对话记忆"
        context_title = "知识库检索内容"
        question_title = "用户问题"

    memory_lines = []
    for item in memory[-10:]:
        role = item.get("role", "")
        text = item.get("text", "")
        if role and text:
            memory_lines.append(f"{role}: {text}")
    memory_text = "\n".join(memory_lines) if memory_lines else ("None" if language == "en" else "暂无")

    context_text = "\n\n---\n\n".join(
        chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk)
        for chunk in context_chunks
    ) or no_context

    base_task = direct_task if mode == "deepseek" else task

    system_prompt = f"""
{base_task}

Language rule:
{lang_rule}

Assistant personality:
{personality_text}

Answer style:
{style_text}

Knowledge base name:
{kb_name}
""".strip()

    user_prompt = f"""
{history_title}:
{memory_text}

{context_title}:
{context_text if mode != "deepseek" else no_context}

{question_title}:
{question}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/history", methods=["GET"])
def history():
    return jsonify({
        "history": load_memory(),
        "state": load_state(),
        "has_knowledge_base": has_knowledge_base(),
    })


@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    save_memory([])
    return jsonify({"message": "memory cleared"})


@app.route("/clear_knowledge", methods=["POST"])
def clear_knowledge():
    clear_vector_db()
    state = load_state()
    state.pop("document_name", None)
    save_state(state)
    return jsonify({"message": "knowledge base cleared"})


@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file received."}), 400

        file = request.files["file"]
        language = normalize_language(request.form.get("language", "zh"))

        if file.filename == "":
            return jsonify({"error": "No file selected." if language == "en" else "文件名为空。"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are supported." if language == "en" else "只支持 PDF 文件。"}), 400

        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        save_path = UPLOAD_FOLDER / unique_filename
        file.save(save_path)

        text = extract_text_from_pdf(save_path)
        if not text or not text.strip():
            return jsonify({"error": "No readable text was extracted from this PDF." if language == "en" else "PDF 没有提取到文字内容。"}), 400

        chunks = split_text(text)
        if not chunks:
            return jsonify({"error": "PDF text splitting failed." if language == "en" else "PDF 文本切分失败。"}), 400

        result = build_vector_db(chunks, source_name=original_filename)

        state = load_state()
        state["document_name"] = original_filename
        state["chunk_count"] = result.get("chunks", len(chunks))
        save_state(state)

        msg = "PDF uploaded successfully. Knowledge base updated." if language == "en" else "PDF 上传成功，知识库已更新。"
        return jsonify({
            "message": msg,
            "filename": original_filename,
            "saved_as": unique_filename,
            "chunks": len(chunks),
        })

    except Exception as e:
        return jsonify({
            "error": "Upload or knowledge base construction failed.",
            "detail": str(e),
        }), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}

        question = data.get("question", "").strip()
        personality = data.get("personality", "professional")
        answer_style = data.get("answer_style", "standard")
        mode = data.get("mode", "knowledge")
        knowledge_name = data.get("knowledge_name", "")
        language = normalize_language(data.get("language", "zh"))

        if not question:
            msg = "Question cannot be empty." if language == "en" else "问题不能为空。"
            return jsonify({"error": msg}), 400

        memory = load_memory()
        context_chunks = []

        if mode != "deepseek":
            context_chunks = search(question, top_k=4)

        messages = build_messages(
            question=question,
            context_chunks=context_chunks,
            personality=personality,
            answer_style=answer_style,
            mode=mode,
            knowledge_name=knowledge_name,
            language=language,
            memory=memory,
        )

        answer = ask_llm(question, messages=messages)

        memory.append({"role": "user", "text": question})
        memory.append({"role": "ai", "text": answer})
        save_memory(memory)

        return jsonify({
            "answer": answer,
            "mode": mode,
            "knowledge_name": knowledge_name,
            "context_count": len(context_chunks),
            "history": load_memory(),
        })

    except Exception as e:
        return jsonify({
            "error": "Question answering failed.",
            "detail": str(e),
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "upload_folder": str(UPLOAD_FOLDER),
        "has_knowledge_base": has_knowledge_base(),
        "memory_count": len(load_memory()),
    })


if __name__ == "__main__":
    app.run(debug=True)
