import torch
import json

# usage:
# define your own verify function for each dataset here
# we provide the verify function for mmbench as an example

def verify(dataset, probs, data_info):
    correctness = []
    if dataset == "mmbench":
        all_options = ['A', 'B', 'C', 'D']
        questions = []
        with open("data/mmbench.jsonl", "r") as f:
            for line in f:
                questions.append(json.loads(line))
        for i in range(len(questions)):
            answer = questions[i]["answer"]
            gt_idx = all_options.index(answer)
            correctness.append(gt_idx == torch.argmax(probs[i]))
        return torch.tensor(correctness)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")