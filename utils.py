import os
import re
import json
import streamlit as st
import torch
from peft import get_peft_model, LoraConfig

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

def load_example_index(index_path):
    with open(index_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_paper_text(text_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()

@st.cache_resource
def load_model(pt_path):
    base_model_name = "Qwen/Qwen3-0.6B-Base"
    # 1. Load base model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    # 2. Load LoRA state dict 
    checkpoint = torch.load(pt_path, map_location="cpu")
    if 'lora_state_dict' not in checkpoint:
        raise ValueError("Checkpoint does not contain LoRA state dict")

    # 3. Reconstruct LoRA config
    config_dict = checkpoint.get('config', {})
    lora_settings = config_dict.get('advanced', {}).get('lora', {})
    r = lora_settings.get('r', 8)              # Fallback values if not present
    alpha = lora_settings.get('alpha', 16)
    target_modules = lora_settings.get('target_modules', ["q_proj", "v_proj"])
    dropout = lora_settings.get('dropout', 0.1)
    bias = lora_settings.get('bias', "none")

    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=target_modules,
        lora_dropout=dropout, bias=bias, task_type="CAUSAL_LM"
    )

    # 4. Wrap base model with LoRA, load weights
    model = get_peft_model(base_model, lora_config)
    model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
    model.eval()
    return tokenizer, model


def summarize(content, tokenizer, model, title="Untitled", topic="arXiv", max_input=1024, max_output=200):
    prompt = f"SUBREDDIT: r/{topic}\nTITLE: {title}\nPOST: {content}\nTL;DR:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input,
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        summary = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )
        return summary.strip()
    
def parse_example_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Try to match the structured format
    topic_match = re.search(r'^TOPIC:\s*(.*)$', text, flags=re.MULTILINE)
    title_match = re.search(r'^TITLE:\s*(.*)$', text, flags=re.MULTILINE)
    content_match = re.search(r'^CONTENT:\s*([\s\S]*)$', text, flags=re.MULTILINE)

    if topic_match and title_match and content_match:
        return {
            'topic': topic_match.group(1).strip(),
            'title': title_match.group(1).strip(),
            'content': content_match.group(1).strip()
        }
    else:
        # Fallback to academic format
        topic = "arXiv"
        title = os.path.basename(filepath)
        content = text
        return {
            'topic': topic,
            'title': title,
            'content': content
        }

