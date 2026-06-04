import os
from dotenv import load_dotenv
from flask import Flask, request, render_template_string
import requests

# 自动加载 .env 文件
load_dotenv()

# 从 .env 读取 API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

app = Flask(__name__)

# 简单 HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>AI Knowledge Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .container { background: #fff; padding: 2rem; border-radius: 8px; width: 600px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        textarea { width: 100%; height: 150px; padding: 10px; font-size: 14px; }
        button { margin-top: 10px; padding: 10px 20px; font-size: 14px; cursor: pointer; }
        .response { margin-top: 20px; padding: 10px; background: #f0f0f0; border-radius: 5px; white-space: pre-wrap; }
    </style>
</head>
<body>
<div class="container">
    <h2>AI Knowledge Assistant</h2>
    <p>基于 DeepSeek API 的智能问答助手</p>
    <form method="POST">
        <textarea name="question" placeholder="请输入你的问题，例如：什么是RAG?" required>{{ question }}</textarea>
        <button type="submit">发送</button>
    </form>
    {% if response %}
    <div class="response">{{ response }}</div>
    {% endif %}
</div>
</body>
</html>
"""

def ask_deepseek(question: str):
    if not DEEPSEEK_API_KEY:
        return "API Key 未配置，请检查 .env 文件。"

    url = "https://api.deepseek.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 500
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        # 根据 DeepSeek 的返回结构提取答案
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "未返回答案")
        return answer
    except requests.exceptions.RequestException as e:
        return f"请求失败：{e}"
    except Exception as e:
        return f"解析响应失败：{e}"

@app.route("/", methods=["GET", "POST"])
def home():
    question = ""
    response = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            response = ask_deepseek(question)
        else:
            response = "问题不能为空！"
    return render_template_string(HTML_TEMPLATE, question=question, response=response)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)