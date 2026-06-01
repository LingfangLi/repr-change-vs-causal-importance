import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import transformers

_NEW_API = int(transformers.__version__.split('.')[0]) >= 5

MODEL_PATH = "<DATA_ROOT>/fine_tuned_model/llama2-7b-yelp-full"

_dtype_kwarg = {"dtype": torch.bfloat16} if _NEW_API else {"torch_dtype": torch.bfloat16}
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **_dtype_kwarg, device_map="auto")
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("yelp_polarity", split="test").select(range(5))

for sample in dataset:
    prompt = f"Review: {sample['text']}\nSentiment:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full[len(prompt):]
    print(f"Label: {sample['label']}")
    print(f"Generated: [{generated}]")
    print("---")
