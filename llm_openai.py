from openai import OpenAI

class ChatGPT_LLM:
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, question: str, context: list[str], max_tokens=300):
        prompt = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{'\n---\n'.join(context)}\n\n"
            f"Question: {question}\n\nAnswer:\n"
        )

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_tokens
        )

        return response.output_text
