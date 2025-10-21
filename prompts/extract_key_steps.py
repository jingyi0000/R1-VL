from openai import OpenAI
import os
import json
import argparse
from tqdm import tqdm
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def gpt_forward(client, prompt, base64_image=None, temperature=0.9):
    content = [{
                    "type": "text",
                    "text": prompt
                }]
    if base64_image is not None:
        content.append({
            "type": "image_url",
            "image_url":{
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        temperature = temperature
    )

    return completion.choices[0].message.content




client = OpenAI(
        api_key="",        
    )



PROMPT = """Given a reasoning process with structured reasoning steps (### headers), extract the key elements from each step.

- For ### Image Description: Extract at most three key elements such as named entities (objects, labels) and numerical values. Each element should be unique.
- For ### Rationales: Extract the core concepts needed to solve the problem, such as mathematical properties or definitions. Ensure each concept appears only once.
- For ### Step X: Extract essential variables, equations, and relationships that contribute to the final solution. Avoid repeating previously extracted elements.
- Ignore ### Let's think step by step and ### The final answer is.

Retain only essential terms, numbers, and equations exactly as they appear in the text. If a key element is a phrase (not a number or equation), ensure it contains at most three words. **Do not extract the same key element more than once across different steps.** 

Format your response as follows:
"Key Elements:\nImage Description: ["key1", "key2", ...]; Rationales: ["key1", "key2", ...]; Step 1: ["key1", "key2", ...], Step 2: ["key1", "key2", ...]; ...".
{reasoning_process}"""



def run_key(args):

    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = open(output_path, "w")

    with open('data_path.json', 'r') as f:
        data = json.load(f)

    data = get_chunk(data, args.num_chunks, args.chunk_idx) 

    for item in tqdm(data):
        dict_item = {}
        image_path = item['images']
        dict_item['images'] = image_path
        mess = item['messages'][1]['content']

        results  = gpt_forward(client, PROMPT.format(reasoning_process=mess))
        dict_item['key_steps'] = results
        output_file.write(json.dumps(dict_item) + "\n")
        output_file.flush()

    output_file.close()

    # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default='output.jsonl')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    args = parser.parse_args()
    
    run_key(args)
