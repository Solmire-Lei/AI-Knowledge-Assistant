import os
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai import (
    OpenAIError,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError(
        "没有读取到 DEEPSEEK_API_KEY，请检查：\n"
        "1. .env 文件是否在项目根目录\n"
        "2. .env 文件名是否正确，必须是 .env\n"
        "3. .env 内容是否为：DEEPSEEK_API_KEY=你的key"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


def ask_llm(
    question: str,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7,
) -> str:
    """Call DeepSeek chat completion API."""
    if not question.strip() and not messages:
        return "问题不能为空，请重新输入。"

    if messages is None:
        messages = [
            {
                "role": "system",
                "content": system_prompt or "你是一个简洁、准确、适合大学生学习的AI助手。回答要清晰易懂。"
            },
            {
                "role": "user",
                "content": question
            }
        ]

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    except AuthenticationError:
        return "API Key 认证失败。请检查 .env 里的 DEEPSEEK_API_KEY 是否正确，或者是否已经失效。"

    except RateLimitError:
        return "API 请求过于频繁，触发限流。请稍等一会儿再试。"

    except APIConnectionError:
        return "无法连接到 DeepSeek API。请检查网络连接，或者稍后重试。"

    except APIStatusError as e:
        if e.status_code == 402:
            return "API 调用失败：余额不足。你的 DeepSeek 账户当前没有可用余额，需要充值后才能继续调用。"
        return f"API 状态错误：{e.status_code}\n错误信息：{e.message}"

    except OpenAIError as e:
        return f"调用 API 时发生错误。\n错误信息：{str(e)}"

    except Exception as e:
        return f"程序发生未知错误。\n错误信息：{str(e)}"


def main():
    print("AI Knowledge Assistant 已启动。")
    print("输入 q / quit / exit 可以退出程序。")

    while True:
        question = input("\n你：")
        if question.lower() in ["q", "quit", "exit"]:
            print("AI：再见！")
            break
        answer = ask_llm(question)
        print("\nAI：")
        print(answer)


if __name__ == "__main__":
    main()
