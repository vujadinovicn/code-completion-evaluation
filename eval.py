import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 

def generate_prompt(example):
    prompt = "<fim_prefix>" + example['prefix'] + "\n<fim_suffix>\n" + example['suffix'] + "<fim_middle>"
    return prompt

def load_json_data(json_path='splitted_code.json'):
    with open("splitted_code.json", "r") as json_file:
        data = json.load(json_file)
    return data

def get_starcoder_model(device):
    checkpoint = "bigcode/tiny_starcoder_py"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    params = {
        'max_new_tokens': 256,
        'temperature': 0.2,
        'top_k': 50,
        'top_p': 0.1,
        'repetition_penalty': 1.17
    }

    return tokenizer, model, params

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, params = get_starcoder_model(device)
    splitted_code_json = load_json_data()

    all_evaluated_data = []
    evalutaed_data_path = "evaluated_data"

    for i, code in enumerate(splitted_code_json):
        input_text = generate_prompt(code)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        generated = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, **params)
        output = tokenizer.decode(generated[0])

        processed_data = {
            "input": input_text,
            "gt_fim_middle": code['middle'],
            "output": output,
            "predicted_fim_middle": output.split("<fim_middle>")[-1].split("<|endoftext|>")[0]
        }
        all_evaluated_data.append(processed_data)

        with open(os.path.join(evalutaed_data_path, "single", f"code_{i+1}.json"), "w") as json_file:
            json.dump(processed_data, json_file, indent=4)
    
    with open(os.path.join(evalutaed_data_path, "all", f"codes.json"), "w") as json_file:   
            json.dump(all_evaluated_data, json_file, indent=4)