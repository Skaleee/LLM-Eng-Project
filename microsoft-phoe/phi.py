import torch
from transformers import AutoTokenizer, PhimoeConfig, PhimoeForCausalLM
from transformers.utils import logging
from accelerate import init_empty_weights, infer_auto_device_map

# Optional: Helps debugging
logging.set_verbosity_info()
device = torch.device("cuda:0")
# Model path
model_path = "microsoft/Phi-mini-MoE-instruct"

# Load config
config = PhimoeConfig.from_pretrained(model_path)
model = PhimoeForCausalLM.from_pretrained(model_path,config=config,
                                         device_map={"": device},
                                         torch_dtype=torch.float16,
                                         )
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

if "inputs" in globals():
    del inputs

prompt = 'Zentropa has much in common with The Third Man, another noir-like film set among the rubble of postwar Europe. Like TTM, there is much inventive camera work. There is an innocent American who gets emotionally involved with a woman he doesn\'t really understand, and whose naivety is all the more striking in contrast with the natives.<br /><br />But I\'d have to say that The Third Man has a more well-crafted storyline. Zentropa is a bit disjointed in this respect. Perhaps this is intentional: it is presented as a dream/nightmare, and making it too coherent would spoil the effect. <br /><br />This movie is unrelentingly grim--"noir" in more than one sense; one never sees the sun shine. Grim, but intriguing, and frightening.' +' Is the prior text positive or negative? Give us a rating out of 10'
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# prob_routing_temp cannot be lower than about 0.0001. 
# 0.00001 causes cuda assert


with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        use_probabilistic_routing=True,
        prob_routing_temp=0.1,
        #output_router_logits=True,
        #return_dict=True
    )

print(tokenizer.decode(outputs[0]))