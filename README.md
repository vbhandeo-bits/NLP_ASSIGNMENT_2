# CS F429 / IS F429 — Assignment 2 (PS21)

Grammatical Error Correction (GEC) as Sequence-to-Sequence Translation

This repository contains the complete solution for Assignment 2 (PS21) for the Natural Language Processing course. The approach frames grammatical error correction as sequence-to-sequence translation and fine-tunes a T5 model on the `agentlans/grammar-correction` dataset.

## Repository contents
- `Assignment2_NLP_PS21.ipynb` — Full Colab / Jupyter notebook with environment setup, data preprocessing, fine-tuning, evaluation, and a Gradio demo for inference.
- `Assignment2-NLP-PS21.pdf` — Assignment brief and submission document (source of task requirements).
- `LICENSE` — Project license.
- `README.md` — This file (updated to include assignment details and reproduction instructions).

Notebook (Colab) link: https://colab.research.google.com/github/vbhandeo-bits/NLP_ASSIGNMENT_2/blob/main/Assignment2_NLP_PS21.ipynb
Assignment PDF (brief): https://github.com/vbhandeo-bits/NLP_ASSIGNMENT_2/blob/main/Assignment2-NLP-PS21.pdf

---

## Team & Submission Details
- Group No: 93

Team members (as listed in the notebook):
- VAIBHAV BHANDEO — BITS ID: 2025AB05033 — 2025AB05033@wilp.bits-pilani.ac.in
- VAIBHAVI VISHWANATH BADIGER — BITS ID: 2025aa05448 — 2025aa05448@wilp.bits-pilani.ac.in
- V Raj Kumar — BITS ID: 2025aa05606 — 2025aa05606@wilp.bits-pilani.ac.in
- VSSGG Rahul Mugada — BITS ID: 2025AA05610 — 2025AA05610@wilp.bits-pilani.ac.in
- Rohit Vadje — BITS ID: 2025aa05165 — 2025aa05165@wilp.bits-pilani.ac.in

Date: 09-August-2026
Environment used: BITS CSIS Labs / CUDA GPU Cluster (e.g., Tesla T4)

---

## What this project does (high level)
- Loads the `agentlans/grammar-correction` dataset and prepares tokenized inputs/targets.
- Fine-tunes `t5-small` for grammatical error correction with Hugging Face Trainer (Seq2SeqTrainer).
- Applies early stopping and saves the best model to `./gec_t5_best`.
- Demonstrates generation strategies to mitigate over-correction (beam search, repetition penalties, no_repeat_ngram_size, length_penalty).
- Evaluates outputs using BERTScore and classical error metrics (WER, CER).
- Provides a Gradio-based local demo for interactive inference.

---

## Key implementation details (from the notebook)
- Model checkpoint: `t5-small`
- Tokenization: task prefix `"grammar correction: "`, `max_length=128`, padding to max length and labels padded with `-100` so padding is ignored in loss.
- Training arguments (as used in the notebook):
  - output_dir: `./gec_t5_results`
  - eval_strategy: `epoch`
  - save_strategy: `epoch`
  - learning_rate: `3e-4`
  - per_device_train_batch_size: `16`
  - per_device_eval_batch_size: `16`
  - weight_decay: `0.01`
  - num_train_epochs: `5`
  - predict_with_generate: `True`
  - fp16: enabled when GPU is available
  - load_best_model_at_end: `True` (metric: `eval_loss`)
  - early stopping patience: 2 epochs

Example generation function (not executable here, see notebook):

```python
# from notebook: generate_corrected_text
formatted_input = PREFIX + input_text
inputs = tokenizer(formatted_input, return_tensors="pt", max_length=128, truncation=True).to(device)
outputs = model.generate(**inputs, max_length=128, num_beams=4, length_penalty=0.8, repetition_penalty=1.1, no_repeat_ngram_size=3, early_stopping=True)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Evaluation metrics
- BERTScore (for semantic similarity)
- WER (Word Error Rate)
- CER (Character Error Rate)

The notebook demonstrates computing these metrics on validation/test splits and reporting training & validation loss per epoch.

Observed training losses (notebook snapshot):
- Epoch 1 — Train Loss: 0.783942 — Val Loss: 0.721942
- Epoch 2 — Train Loss: 0.736856 — Val Loss: 0.710152
- Epoch 3 — Train Loss: 0.664863 — Val Loss: 0.711411
- Epoch 4 — Train Loss: 0.642388 — Val Loss: 0.709687

The best checkpoint is saved to `./gec_t5_best` in the notebook's working directory.

---

## Reproduce / Run instructions
1. Recommended: run on a CUDA-enabled machine (GPU like T4); Colab is supported via the notebook link above.
2. From a Python environment with pip, install dependencies used in the notebook:

```bash
pip install transformers datasets evaluate bert-score jiwer gradio torch accelerate
```

3. Open `Assignment2_NLP_PS21.ipynb` in Colab (link above) or locally with Jupyter and run cells in order. The notebook handles dataset download, preprocessing, training, evaluation, and saving the model.

4. For inference, load the saved model directory from the notebook (`./gec_t5_best`) with `AutoTokenizer.from_pretrained("./gec_t5_best")` and `AutoModelForSeq2SeqLM.from_pretrained("./gec_t5_best")` and use the generation routine shown earlier.

Notes:
- The notebook may download model files from Hugging Face and dataset files; set `HF_TOKEN` environment variable if you encounter rate limits when downloading from the hub.
- Training may require substantial GPU memory. Reduce batch size or use gradient accumulation if running into OOM.

---

## Files to inspect
- `Assignment2_NLP_PS21.ipynb` — primary implementation and narrative. Start here to reproduce experiments.
- `Assignment2-NLP-PS21.pdf` — assignment brief; verifies required deliverables for PS21.

---

## License
See `LICENSE`.

---

## Contact
For questions about this submission, contact the group lead:
- VAIBHAV BHANDEO — 2025AB05033@wilp.bits-pilani.ac.in

