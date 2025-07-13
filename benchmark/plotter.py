import os
import json
import matplotlib.pyplot as plt

# --- Locate JSON relative to script ---
script_dir = os.path.dirname(os.path.abspath(__file__))
json_file = os.path.join(script_dir, "moe_eval_results_old.json")

print("Using JSON path:", json_file)

# --- Ensure JSON file exists ---
if not os.path.exists(json_file):
    raise FileNotFoundError(f"JSON file not found: {json_file}")

# --- Load data ---
with open(json_file, "r") as f:
    data = json.load(f)

# --- Iterate through models ---
for model_name, results in data.items():
    # Convert string keys to (key_str, float_val), then sort by float
    temp_pairs = [(k, float(k)) for k in results.keys()]
    temp_pairs.sort(key=lambda x: x[1])
    temperature_keys, temperatures = zip(*temp_pairs)

    perplexities = []
    bleu_scores = []
    rouge1_scores = []

    for temp_key in temperature_keys:
        entry = results[temp_key]
        perplexities.append(entry["Perplexity"])
        bleu_scores.append(entry["Translation"]["BLEU"])
        rouge1_scores.append(entry["Summarization"]["rouge1"])

    # Clean model name for file naming
    clean_model_name = model_name.replace("/", "_")

    # --- Plot Perplexity ---
    plt.figure()
    plt.plot(temperatures, perplexities, marker="o")
    plt.title(f"{model_name} - Perplexity vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Perplexity")
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, f"{clean_model_name}_perplexity.png"))
    plt.close()

    # --- Plot BLEU ---
    plt.figure()
    plt.plot(temperatures, bleu_scores, marker="o", color="green")
    plt.title(f"{model_name} - Translation BLEU vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("BLEU Score")
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, f"{clean_model_name}_translation_bleu.png"))
    plt.close()

    # --- Plot ROUGE-1 ---
    plt.figure()
    plt.plot(temperatures, rouge1_scores, marker="o", color="red")
    plt.title(f"{model_name} - Summarization ROUGE-1 vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("ROUGE-1 Score")
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, f"{clean_model_name}_summarization_rouge1.png"))
    plt.close()

    print(f"✅ Saved plots for: {model_name}")

print("🎉 All plots saved successfully.")
