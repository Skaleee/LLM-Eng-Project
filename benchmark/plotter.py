import os
import json
import matplotlib.pyplot as plt

# --- List of JSON filenames ---
json_files = [
    "moe_eval_results.json",
    "moe_eval_results_old.json",
    "moe_eval_results100.json",
    "moe_eval_results400.json"
]

# --- Get base path ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Loop through each file ---
for json_file in json_files:
    file_path = os.path.join(script_dir, json_file)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {json_file}")
        continue

    # --- Load data ---
    with open(file_path, "r") as f:
        data = json.load(f)

    # --- Output subfolder ---
    folder_name = f"plots_{os.path.splitext(json_file)[0]}"
    output_dir = os.path.join(script_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 Processing: {json_file} → {folder_name}")

    # --- Loop through each model ---
    for model_name, results in data.items():
        temp_pairs = [(k, float(k)) for k in results.keys()]
        temp_pairs.sort(key=lambda x: x[1])
        temperature_keys, temperatures = zip(*temp_pairs)

        # Initialize metric lists
        perplexities = []
        bleu_scores = []
        meteor_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougel_scores = []
        rougelsum_scores = []

        for temp_key in temperature_keys:
            entry = results[temp_key]
            perplexities.append(entry["Perplexity"])
            bleu_scores.append(entry["Translation"]["BLEU"])
            meteor_scores.append(entry["Translation"]["METEOR"])
            rouge1_scores.append(entry["Summarization"]["rouge1"])
            rouge2_scores.append(entry["Summarization"]["rouge2"])
            rougel_scores.append(entry["Summarization"]["rougeL"])
            rougelsum_scores.append(entry["Summarization"]["rougeLsum"])

        clean_model_name = model_name.replace("/", "_")

        # --- Helper to plot any metric ---
        def plot_metric(values, label, ylabel, color, suffix, round_fmt):
            plt.figure()
            plt.plot(temperatures, values, marker="o", color=color)
            if "0" in results:
                # Fetch baseline safely from the correct structure
                if label == "Perplexity":
                    baseline = results["0"]["Perplexity"]
                elif label == "BLEU":
                    baseline = results["0"]["Translation"]["BLEU"]
                elif label == "METEOR":
                    baseline = results["0"]["Translation"]["METEOR"]
                else:
                    baseline = results["0"]["Summarization"][suffix]
                plt.axhline(y=baseline, color="gray", linestyle="--", linewidth=1,
                            label=f'No Routing: {round_fmt.format(baseline)}')
                plt.legend()
            plt.title(f"{model_name} - {label} vs Temperature")
            plt.xlabel("Temperature")
            plt.ylabel(ylabel)
            plt.grid(True)
            filename = f"{clean_model_name}_{suffix.lower()}.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

        # --- Generate all plots ---
        plot_metric(perplexities, "Perplexity", "Perplexity", "blue", "Perplexity", "{:.2f}")
        plot_metric(bleu_scores, "BLEU", "BLEU Score", "green", "BLEU", "{:.2f}")
        plot_metric(meteor_scores, "METEOR", "METEOR Score", "teal", "METEOR", "{:.3f}")
        plot_metric(rouge1_scores, "ROUGE-1", "ROUGE-1 Score", "red", "rouge1", "{:.3f}")
        plot_metric(rouge2_scores, "ROUGE-2", "ROUGE-2 Score", "orange", "rouge2", "{:.3f}")
        plot_metric(rougel_scores, "ROUGE-L", "ROUGE-L Score", "purple", "rougeL", "{:.3f}")
        plot_metric(rougelsum_scores, "ROUGE-Lsum", "ROUGE-Lsum Score", "brown", "rougeLsum", "{:.3f}")

        print(f"✅ Plots saved for model: {model_name}")

print("🎉 All files processed and all metrics plotted.")
