import json
from typing import List, Dict, Any

# always use type hinting


# only load up to500 samples
with open('data/gsm8k/test.jsonl', 'r') as f:
    samples = f.readlines()[:500]

CoT_prompt = "Let's think step by step. At the end, you MUST write the answer as an integer after '####'."

def query_llama(sample: str) -> str:


def test_sample(sample: str, answer: str) -> bool:
    answer_num: int = int(answer.split("####")[-1].strip())
    


