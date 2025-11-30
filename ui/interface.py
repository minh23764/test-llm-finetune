from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR = Path(__file__).parent.parent
ADAPTER = BASE_DIR / "outputs" / "phi3_minh_lora"
BASE = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE).to("cuda")
model = PeftModel.from_pretrained(base, ADAPTER).to("cuda")
model.eval()


def ask(q):
    prompt = f"User: {q}\nAssistant:"
    tokens = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **tokens,
        max_new_tokens=120,
        do_sample=False,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


ask("Who are you?")
ask("Who is Minh?")
ask("Is Mai fat?")
ask("Explain OOP in C#")
