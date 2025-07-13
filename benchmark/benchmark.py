import torch
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM, OlmoeForCausalLM, OlmoeConfig, PhimoeConfig, PhimoeForCausalLM
from transformers.utils import logging
from accelerate import init_empty_weights
from datasets import load_dataset
import evaluate
from tqdm import tqdm
from torch.utils.data import DataLoader
import gc
import json
from collections import defaultdict
import numpy as np
import os


def evaluation_loop(model, tokenizer, datasets, temperatures, results, model_name):
    print(model_name)
    for temperature in temperatures:
        print(f'Temperature: {temperature}')
        perplex = evaluate_perplexity(model, tokenizer, datasets[0], True, temperature, model_name)
        trans_metrics = evaluate_translation(model, tokenizer ,datasets[1], True, temperature, model_name)
        summ_metrics = evaluate_summary(model, tokenizer ,datasets[2], True, temperature, model_name)

        # Store all results
        results[model_name][temperature]["Perplexity"] = perplex
        results[model_name][temperature]["Translation"] = trans_metrics
        results[model_name][temperature]["Summarization"] = summ_metrics

    perplex = evaluate_perplexity(model, tokenizer, datasets[0], False, temperature, model_name)
    trans_metrics = evaluate_translation(model, tokenizer ,datasets[1], False, temperature, model_name)
    summ_metrics = evaluate_summary(model, tokenizer ,datasets[2], False, temperature, model_name)

    results[model_name][0]["Perplexity"] = perplex
    results[model_name][0]["Translation"] = trans_metrics
    results[model_name][0]["Summarization"] = summ_metrics

# def evaluate_perplexity(model, tokenizer, dataset, routing, temp, model_name):
#     log_likelihoods = []

#     for sample in dataset.select(range(10)):  # subset for speed
#         input_text = sample["text"].strip()
#         if not input_text:
#             continue  # skip empty lines

#         inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

#         # Shift inputs for loss computation
#         input_ids = inputs["input_ids"]
#         with torch.no_grad():
#             outputs = model(input_ids=input_ids, labels=input_ids, use_probabilistic_routing=routing, prob_routing_temp=temp)
#             loss = outputs.loss
#         log_likelihoods.append(loss.item())

#     #Compute perplexity
#     perplexity = torch.exp(torch.tensor(log_likelihoods).mean()).item()
#     print(f"Perplexity: {perplexity:.2f}")

from torch.utils.data import DataLoader

def evaluate_perplexity(model, tokenizer, dataset, routing, temp, model_name, batch_size=32):
    model.eval()
    log_likelihoods = []

    # Subset for speed
    subset = dataset.select(range(400))  # or 10 if you want parity

    # Prepare dataloader
    # Tokenize all prompts in batch
    if type(model) is Qwen3MoeForCausalLM:
        dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            collate_fn=lambda batch: tokenizer(
                [sample["text"].strip() for sample in batch if sample["text"].strip()],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
        )
    else:
        dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            collate_fn=lambda batch: tokenizer(
                [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": sample["text"].strip()} for sample in batch if sample["text"].strip()],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    for sample in batch if sample["text"].strip()
                ],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
        )

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_probabilistic_routing=routing,
                prob_routing_temp=temp,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True
            )
            # Compute per-sample loss: shape (batch_size,)
            # HuggingFace models return `loss` averaged over non-masked tokens by default.
            # We recompute per-sample losses below:
            logits = outputs.logits  # (B, T, V)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            # Compute per-token NLL loss (no reduction)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            vocab_size = shift_logits.size(-1)
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            ).view(shift_labels.size())

            # Mask and compute mean loss per sample
            per_sample_loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)

        log_likelihoods.extend(per_sample_loss.tolist())

    # save(model_name, 'preplexity', temp, input_ids, labels, outputs.logits)
    
    # Final perplexity
    mean_nll = torch.tensor(log_likelihoods).mean()
    perplexity = torch.exp(mean_nll).item()
    print(f"Perplexity: {perplexity:.2f}")
    return perplexity


def evaluate_translation(model, tokenizer, dataset, routing, temp, model_name, batch_size=32):
    references, predictions = [], []

    # Select a subset and prepare prompts
    subset = dataset["train"].select(range(400))
    prompts = [f"{ex['instruction']}\n{ex['input']}" for ex in subset]
    gold_outputs = [ex['output'] for ex in subset]

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Tokenize all prompts in batch
    if type(model) is Qwen3MoeForCausalLM:
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    else:
        messages = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for prompt in prompts]
        tokenized = tokenizer(messages, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Create DataLoader manually (we do batching ourselves here)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    n_samples = input_ids.size(0)

    for i in tqdm(range(0, n_samples, batch_size), desc="Evaluating Translation"):
        batch_input_ids = input_ids[i:i+batch_size].to(model.device)
        batch_attention_mask = attention_mask[i:i+batch_size].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=100,
                use_probabilistic_routing=routing,
                prob_routing_temp=temp,
            )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_refs = gold_outputs[i:i+batch_size]
        tokenized_refs = tokenizer(decoded_refs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        labels = tokenized_refs["input_ids"]

        predictions.extend(decoded_preds)
        references.extend([[ref] for ref in decoded_refs])  # BLEU/TER expects list of lists

    # save(model_name, 'translation', temp, input_ids, labels, outputs)
    
    # Compute evaluation metrics
    bleu_score = bleu.compute(predictions=predictions, references=references)["score"]
    meteor_score = meteor.compute(predictions=predictions, references=[r[0] for r in references])["meteor"]

    print(f"\n--- Translation Evaluation @ Temp = {temp} ---")
    print("BLEU:", bleu_score)
    print("METEOR:", meteor_score)
    # print("TER:", ter.compute(predictions=predictions, references=references))
    return {"BLEU": bleu_score, "METEOR": meteor_score}



def evaluate_summary(model, tokenizer, dataset, routing, temp, model_name, batch_size=32):
    # Make sure padding is on the left for decoder-only models
    tokenizer.padding_side = "left"

    # Select a subset and convert to HuggingFace Dataset if not already
    data = dataset["train"].select(range(400))

    references = []
    predictions = []

    # Prepare batches
    dataloader = DataLoader(data, batch_size=batch_size)

    for batch in tqdm(dataloader, desc=f"Evaluating summaries @ Temp {temp}"):
        texts = batch["text"]
        targets = batch["summarization"]

        prompts = [f"Summarize the following text such that it is semantically correct: {text}" for text in texts]

        # Tokenize all prompts in batch
        if type(model) is Qwen3MoeForCausalLM:
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
        else:
            messages = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in prompts]
            inputs = tokenizer(
                messages,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

        # Generate summaries
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            use_probabilistic_routing=routing,
            prob_routing_temp=temp,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        references.extend(targets)
        predictions.extend(decoded)
        tokenized_refs = tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=512)
        labels = tokenized_refs["input_ids"]

    # save(model_name, 'summary', temp, inputs['input_ids'], labels, outputs)

    # Compute metrics
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    print(f"\n--- Summary Evaluation @ Temp = {temp} ---")
    print("ROUGE:", rouge_scores)
    return rouge_scores

def save(model_name, eval, temp, input, label, output):
    filename = f"{model_name}-{eval}-{temp}.npz"
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.savez(
        f"{model_name}-{eval}-{temp}.npz",
        input_ids=input.to(torch.float32).cpu().numpy(),
        labels=label.to(torch.float32).cpu().numpy(),
        outputs=output.to(torch.float32).cpu().numpy()
    )

results = defaultdict(lambda: defaultdict(dict))

perplexity = evaluate.load("perplexity", module_type="metric")
bleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")
ter = evaluate.load("ter")
rouge = evaluate.load("rouge")
bert_score = evaluate.load("bertscore")

# wikitext = load_dataset("Salesforce/wikitext", 'wikitext-103-raw-v1')
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
translations = load_dataset("mfmezger/sandboxai_german_to_english_translations_seperated")
summaries = load_dataset("ProCreations/simple-summaries")
datasets = [wikitext, translations, summaries]
print("Datasets loaded ...")

T = [0.01, 0.1, 0.3, 0.5, 0.7, 1, 5]

# Model path
model_path = "Qwen/Qwen3-30B-A3B"

# Load and evaluate qwen3moe
config = Qwen3MoeConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = Qwen3MoeForCausalLM.from_pretrained(
    model_path,
    config=config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

evaluation_loop(model, tokenizer, datasets, T, results, model_path)

del model 
gc.collect()
torch.cuda.empty_cache()

# Load and evaluate olmoe
torch.set_printoptions(profile='simple')
device = torch.device("cuda:0")
model_path = "allenai/OLMoE-1B-7B-0924-Instruct" #"allenai/OLMoE-1B-7B-0924" 
config = OlmoeConfig.from_pretrained(model_path,num_experts_per_tok=4)
model = OlmoeForCausalLM.from_pretrained(
    model_path,
    config=config,
    device_map={"": device},
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

evaluation_loop(model, tokenizer, datasets, T, results, model_path)

del model
gc.collect()
torch.cuda.empty_cache()

# Load and evaluate phi moe
model_path = "microsoft/Phi-3.5-MoE-instruct"
config = PhimoeConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = PhimoeForCausalLM.from_pretrained(
    model_path,
    config=config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

evaluation_loop(model, tokenizer, datasets, T, results, model_path)

# Save results to JSON file
with open("moe_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
