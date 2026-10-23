import os
from typing import Optional


class GroqGenerator:
    """
    Hosted instruction-tuned LLM via Groq (reads GROQ_API_KEY from env / .env).
    Follows citation instructions far more reliably than flan-t5-base.
    """
    supports_system = True

    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.0):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key or key == "your_key_here":
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = Groq(api_key=key)
        self.model_name = model_name
        self.temperature = temperature

    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 600,
        system: Optional[str] = None
    ) -> str:
        messages = []
    
        if system:
            messages.append({
                "role": "system",
                "content": system
            })
    
        messages.append({
            "role": "user",
            "content": prompt
        })
    
        
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=max_new_tokens,
            reasoning_effort="low",
        )

    
        choice = resp.choices[0]
        content = choice.message.content or ""
    
        print("\n--- GENERATOR DEBUG ---")
        print("Model:", self.model_name)
        print("Finish reason:", choice.finish_reason)
        print("Response content:", repr(content))
        print("Usage:", resp.usage)
        print("-----------------------\n")
    
        return content.strip()


class LocalGenerator:
    """Offline fallback (flan-t5). Weak at following citation instructions; use for smoke tests only."""
    supports_system = False

    def __init__(self, model_name: str = "google/flan-t5-base"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def generate(self, prompt: str, max_new_tokens: int = 300, system: Optional[str] = None) -> str:
        if system:
            prompt = f"{system}\n\n{prompt}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def get_generator(prefer: str = "groq"):
    """Groq if a key is configured, otherwise the local model."""
    if prefer == "groq":
        try:
            return GroqGenerator()
        except Exception as e:  # missing key / package
            print(f"[generator] Groq unavailable ({e}); falling back to local flan-t5.")
    return LocalGenerator()
