import json
from typing import List, Dict, Any
from gradio_client import Client
import re

client = Client("https://576419700ef8950c0e.gradio.live/")

# always use type hinting
# only load up to 500 samples

with open('results/gsm8k/llama3_8b_instruct/stats.json', 'r') as f:
    current_sample: int = json.load(f)["current_sample"]

with open('data/gsm8k/test.jsonl', 'r') as f:
    samples: List[Dict[str, Any]] = [json.loads(line) for line in f.readlines()][current_sample: 500]

CoT_prompt = "Let's think step by step. At the end, you MUST write the answer as an integer after '####'."

def query_llama(sample: str) -> str:
    result = client.predict(
        api_name="/chat",
        message=sample
    )
    return result

def test_sample(sample: str, answer: str) -> None:
    answer_num: int = int(answer.split("####")[-1].strip())
    llama_answer: str = query_llama(sample + "\n\n" + CoT_prompt)
    print(llama_answer)
    # split by ####, strip and regex out the int 
    correct: bool = int(re.findall(r'\d+', llama_answer.split("####")[-1].strip())[0]) == answer_num

    # save sample
    with open("results/gsm8k/llama3_8b_instruct/full.jsonl", "a") as f:
        # the file is an array of objects
        f.write(json.dumps({"sample": sample, "answer": answer, "correct": correct}) + "\n")
    
    with open("results/gsm8k/llama3_8b_instruct/stats.json", "r") as f:
        stats = json.load(f)
        # recompute stats
        stats["total_samples"] += 1
        if correct:
            stats["correct_samples"] += 1
        else:
            stats["incorrect_samples"] += 1
        stats["accuracy"] = 100 * stats["correct_samples"] / stats["total_samples"]
        stats["current_sample"] += 1
        print("\n\n")
        print(stats)
        with open("results/gsm8k/llama3_8b_instruct/stats.json", "w") as f:
            json.dump(stats, f)



for sample in samples:
    test_sample(sample["question"], sample["answer"])