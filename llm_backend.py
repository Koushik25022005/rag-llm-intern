from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LocalLLM:
    def __init__(self, model_name='google/flan-t5-small', device='cpu'):
        self.model_name = model_name
        self.device = device
        self._init_model()

    def _init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if self.device != 'cpu' and torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.pipe = pipeline(
            'text2text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if (self.device!='cpu' and torch.cuda.is_available()) else -1
        )

    def generate(self, question: str, context: list[str], max_length=256):
        prompt = (
            "Context:\n" + "\n---\n".join(context) +
            f"\n\nQuestion: {question}\nAnswer:"
        )
        result = self.pipe(prompt, max_length=max_length, truncation=True)[0]
        return result['generated_text']
