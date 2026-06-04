import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai import (
    OpenAIError,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)


# 1. 获取当前文件所在目录
BASE_DIR = Path(__file__).resolve().parent

# 2. 指定 .env 文件路径
ENV_PATH = BASE_DIR / ".env"

# 3. 加载 .env 文件
load_dotenv(dotenv_path=ENV_PATH, override=True)

# 4. 读取 DeepSeek API Key
api_key = os.getenv("DEEPSEEK_API_KEY")

# 5. 检查 API Key 是否读取成功
if not api_key:
    raise ValueError(
        "没有读取到 DEEPSEEK_API_KEY，请检查：\n"
        "1. .env 文件是否在项目根目录\n"
        "2. .env 文件名是否正确，必须是 .env\n"
        "3. .env 内容是否为：DEEPSEEK_API_KEY=你的key"
    )


# 6. 创建 DeepSeek 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


def ask_llm(question: str) -> str:
    """
    调用 DeepSeek API，输入问题，返回模型回答。
    """

    if not question.strip():
        return "问题不能为空，请重新输入。"

    try:
        response = client.chat.completions.create(
           model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个简洁、准确、适合大学生学习的AI助手。回答要清晰易懂。"
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    except AuthenticationError:
        return (
            "API Key 认证失败。\n"
            "请检查 .env 里的 DEEPSEEK_API_KEY 是否正确，或者是否已经失效。"
        )

    except RateLimitError:
        return (
            "API 请求过于频繁，触发限流。\n"
            "请稍等一会儿再试。"
        )

    except APIConnectionError:
        return (
            "无法连接到 DeepSeek API。\n"
            "请检查网络连接，或者稍后重试。"
        )

    except APIStatusError as e:
        if e.status_code == 402:
            return (
                "API 调用失败：余额不足。\n"
                "你的 DeepSeek 账户当前没有可用余额，需要充值后才能继续调用。"
            )

        return (
            f"API 状态错误：{e.status_code}\n"
            f"错误信息：{e.message}"
        )

    except OpenAIError as e:
        return (
            "调用 API 时发生错误。\n"
            f"错误信息：{str(e)}"
        )

    except Exception as e:
        return (
            "程序发生未知错误。\n"
            f"错误信息：{str(e)}"
        )


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