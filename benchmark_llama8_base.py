import json
from typing import List, Dict, Any
from gradio_client import Client

client = Client("https://576419700ef8950c0e.gradio.live/")

# always use type hinting
# only load up to 500 samples

with open('data/gsm8k/test.jsonl', 'r') as f:
    samples = f.readlines()[:2]

CoT_prompt = "Let's think step by step. At the end, you MUST write the answer as an integer after '####'."

def query_llama(sample: str) -> str:
    result = client.predict(
        message=sample,
        api_name="/chat"
    )
    return result

def test_sample(sample: str, answer: str) -> None:
    answer_num: int = int(answer.split("####")[-1].strip())
    llama_answer: str = query_llama(sample)
    correct: bool = int(llama_answer.split["####"][-1].strip()) == answer_num

    # save sample
    with open("results/gsm8k_llama3_8b_instruct_base.jsonl", "a") as f:
        # the file is an array of objects
        f.write(json.dumps({"sample": sample, "answer": answer, "correct": correct}) + "\n")
    
    


