import json
from pathlib import Path
from typing import List

def default_few_shot_prompt_generation(BATCH_SIZE: int = 5) -> List[str]:
    """
    Generates batched few-shot prompts for QA generation.
    Much of this is hardcoded because it processes a specific (default) example dataset.

    Returns:
        List[str]: A list of prompt strings, one per batch.
    """

    # Get project root (formal-science/)
    ROOT = Path(__file__).resolve().parents[2]

    # Obtain few-shot QA examples
    raw_file_path = ROOT / "legacy" / "raw_text.txt"
    with open(raw_file_path, "r") as f:
        text = f.read().replace(" }","}").replace("}:",":}")
    parts = text.split("\n\n--------------------------------------------------------------------------------------------------------------------------------------------------\n\n")
    few_shot_data = []
    for i in range(0, len(parts) - 1, 2):
        few_shot_data.append({
            "question": parts[i],
            "answer": parts[i+1]
        })

    # Obtain informal reasoning examples from physics derivations (from EMNLP paper)
    physics_file_path = ROOT / "legacy" / "physics_derivations.json"
    with open(physics_file_path,"r") as f:
        physics_data = json.load(f)

    # We split derivations into batches for fewer prompts (and hence less human iterations)
    if (BATCH_SIZE < 1) or (BATCH_SIZE > len(physics_data)):
        raise ValueError("BATCH_SIZE is larger than number of derivations or less than 1.") 

    n = len(physics_data) // BATCH_SIZE # Ok to drop data
    batched_data = []
    for i in range(n):
        j = i*BATCH_SIZE
        batched_data.append(physics_data[j:j+BATCH_SIZE])

    # few-shot QA generation preamble
    init_prompt = "The following 5 questions (Q1-Q5) and respective answers (A1-A5) are few-shot examples. These are given below as: \n\n"
    for example in few_shot_data:
        q, a = example["question"], example["answer"]
        init_prompt += f"{q}\n\n{a}\n\n\n\n"

    # 1 var_prompt per 5 derivations from batched_data
    context_prompts = []
    last_idx = len(few_shot_data) + BATCH_SIZE
    for batch in batched_data:
        batch_prompt = (
            f"Now, the following **equation-only** derivations (D6-D{last_idx}) represent "
            "the underlying equational reasoning of a Physics derivation. You must convert "
            f"each derivation into a **physically-correct** and **contextually-enriched** "
            f"Question (Q6-Q{last_idx}) and Answer (A6-A{last_idx}) pair closely based on "
            "the previous few-shot examples. Each question must be fully self-contained and "
            "must not depend on previous questions, previous results, or external context not "
            "stated in that question:\n\n"
        )
        var_prompt = init_prompt + batch_prompt
        for i in range(BATCH_SIZE):
            example = batch[i]
            derivation = example["derivation"].replace("and ","\n\n")

            var_prompt += f"D{i + len(few_shot_data) + 1}: {derivation}\n\n"
        var_prompt += (
            "Ensure that you write only one equality per equation, ensure correct physical "
            "meaning, use **standard notation** throughout, and make every question "
            "**self-contained**."
        )
        context_prompts.append(var_prompt)

    return context_prompts
