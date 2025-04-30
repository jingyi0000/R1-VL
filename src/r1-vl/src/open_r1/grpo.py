# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from PIL import Image

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "validity"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )



def check_reasoning_completeness(text):
    """
    Check if the text contains all required sections.
    """
    required_sections = [
        "### Image Description:",
        "### Rationales:",
        "### Step 1:",
        "### The final answer is:"
    ]
    
    missing_sections = [section for section in required_sections if section not in text]
    return not missing_sections

def check_reasoning_logic(text):
    """
    Check if the sections appear in the correct order.
    """
    # Ensure `### The final answer is:` appears last
    if not text.strip().endswith("### The final answer is:"):
        final_answer_match = re.search(r"### The final answer is:.*?$", text, re.DOTALL)
        if not final_answer_match or final_answer_match.start() < max(re.finditer(r"### Step \d+:", text), key=lambda x: x.end()).end():
            return False
    
    # Ensure `### Step X:` exists and appears in correct order
    step_matches = re.findall(r"### Step (\d+):", text)
    if not step_matches:
        return False
    
    step_numbers = list(map(int, step_matches))
    return step_numbers == list(range(1, len(step_numbers) + 1))

def validity_reward(completions, **kwargs):

    completion_contents = [completion[0]["content"] for completion in completions]
    return [1.0 if (check_reasoning_completeness(content) and check_reasoning_logic(content)) else 0.0 for content in completion_contents]



def key_step_single(content, key_step):

    step_texts = re.findall(r"### Step \d+:\s*(.*?)(?=\n###|\Z)", content, re.DOTALL)
    step_text_combined = " ".join(step_texts).lower()

    steps_keywords = key_step.get("Steps", [])
    total_keywords = len(steps_keywords)

    matched_keywords = 0
    for group in steps_keywords:
        if isinstance(group, str):
            group = [group]
        if any(alt.lower() in step_text_combined for alt in group):
            matched_keywords += 1

    reward = matched_keywords / total_keywords if total_keywords > 0 else 0.0
    return reward


def accuracy_reward(completions, solution, key_steps, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    # for content, sol in zip(contents, solution):
    for content, sol, key_step in zip(contents, solution, key_steps):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                # sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                sol_match = re.search(r'### The final answer is:\s*\n?(.*)', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                # content_match = re.search(r'<answer>(.*?)</answer>', content)
                content_match = re.search(r'### The final answer is:\s*\n?(.*)', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        if key_steps is not None:
            step_r = key_step_single(content, key_step)
            reward += step_r * 0.1

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "validity": validity_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = load_dataset('json', data_files={'train': script_args.dataset_name})
    

    def load_images(example):

        image = Image.open(example["image_path"]).convert("RGBA")
        example["image"] = image
        
        return example
    
    dataset = dataset.map(load_images)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }


    # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
