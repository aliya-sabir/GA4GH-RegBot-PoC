import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def main() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not found in environment or .env")

    model = os.getenv("HF_MODEL", DEFAULT_MODEL)
    client = InferenceClient(model=model, token=token)

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a concise test assistant."},
                {"role": "user", "content": "Reply with: LLM OK"},
            ],
            max_tokens=10,
            temperature=0.0,
        )
    except Exception as exc:
        raise SystemExit(f"LLM request failed: {exc}")

    content = response.choices[0].message.content
    print(f"Model: {model}")
    print(f"Response: {content}")


if __name__ == "__main__":
    main()
