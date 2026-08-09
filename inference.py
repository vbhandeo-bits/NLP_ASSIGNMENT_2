# inference.py
# Minimal example to load the fine-tuned model saved by the notebook at ./gec_t5_best
# Usage: python inference.py "This is an example sentence with eror."

import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_DIR = "./gec_t5_best"
PREFIX = "grammar correction: "


def load_model(model_dir=MODEL_DIR, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model, device


def correct_sentence(sentence, tokenizer, model, device, max_length=128):
    inputs = tokenizer(PREFIX + sentence, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            length_penalty=0.8,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py \"Your sentence here.\"")
        return
    sentence = sys.argv[1]
    tokenizer, model, device = load_model()
    corrected = correct_sentence(sentence, tokenizer, model, device)
    print("Input:   ", sentence)
    print("Corrected:", corrected)


if __name__ == "__main__":
    main()
