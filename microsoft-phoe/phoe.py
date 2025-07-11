import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "microsoft/Phi-mini-MoE-instruct"

# === Load tokenizer and model ===
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

model.eval()

# === Load small eval dataset (wikitext) ===
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:1%]")

max_length = 64
def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=max_length, padding="max_length")

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

batch_size = 4

# === Routing controller to patch model's router layers ===
def make_prob_forward(original_forward, temperature):
    def forward(x):
        logits = original_forward(x)
        probs = torch.softmax(logits / temperature, dim=-1)
        batch, n_experts = probs.shape
        indices = torch.multinomial(probs.view(-1, n_experts), 1).view(batch)
        probs = F.one_hot(indices, num_classes=n_experts).float()
        return probs
    return forward

def make_top1_forward(original_forward):
    def forward(x):
        logits = original_forward(x)
        probs = torch.softmax(logits, dim=-1)
        batch, n_experts = probs.shape
        indices = torch.argmax(probs, dim=-1)
        probs = F.one_hot(indices, num_classes=n_experts).float()
        return probs
    return forward

class RoutingControllerSimple:
    def __init__(self, model, temperature=.1):
        self.model = model
        self.temperature = temperature
        self.router_layers = []
        self.original_forwards = {}

        for name, module in model.named_modules():
            if ("router" in name or "gate" in name) and isinstance(module, torch.nn.Linear):
                module._name = name
                self.router_layers.append((name, module))

        print(f"Found {len(self.router_layers)} router layers:")
        for name, module in self.router_layers:
            print(" ", name, module)
            
    def patch_probabilistic(self):
        for name, router in self.router_layers:
            self.original_forwards[name] = router.forward
            router.forward = make_prob_forward(router.forward, self.temperature)

    def patch_top1(self):
        for name, router in self.router_layers:
            self.original_forwards[name] = router.forward
            router.forward = make_top1_forward(router.forward)

    def restore(self):
        for name, router in self.router_layers:
            if name in self.original_forwards:
                router.forward = self.original_forwards[name]


# === Evaluation loop ===
def evaluate(model, dataloader, routing_controller=None):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            losses.append(loss.item())

    if routing_controller:
        routing_controller.restore()
    return sum(losses) / len(losses)


# === Prepare dataloader ===
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size)

# === Run evaluation for top1 routing ===
routing_controller = RoutingControllerSimple(model, temperature=0.1)

print("\nEvaluating with Top-1 Routing...")
routing_controller.patch_top1()
loss_top1 = evaluate(model, dataloader, routing_controller)
routing_controller.restore()
print(f"Loss (Top-1 routing): {loss_top1:.4f}")
print(f"Perplexity (Top-1 routing): {torch.exp(torch.tensor(loss_top1)):.4f}")

# === Run evaluation for probabilistic routing ===
print("\nEvaluating with Probabilistic Routing...")
routing_controller.patch_probabilistic()
loss_prob = evaluate(model, dataloader, routing_controller)
routing_controller.restore()
print(f"Loss (Probabilistic routing): {loss_prob:.4f}")
print(f"Perplexity (Probabilistic routing): {torch.exp(torch.tensor(loss_prob)):.4f}")
