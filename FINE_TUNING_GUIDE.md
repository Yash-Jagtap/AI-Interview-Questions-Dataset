# LLM Fine-Tuning Guide for Interview Simulation

This guide covers how to fine-tune open-source LLMs to create an AI interviewer that can simulate realistic coding interviews, provide problems, evaluate solutions, and offer constructive feedback.

## Table of Contents
1. [Overview](#overview)
2. [Choosing the Right Model](#choosing-the-right-model)
3. [Environment Setup](#environment-setup)
4. [Data Preparation](#data-preparation)
5. [Fine-Tuning Process](#fine-tuning-process)
6. [Creating an Interview Simulator](#creating-an-interview-simulator)
7. [Testing & Benchmarks](#testing--benchmarks)
8. [Deployment with Ollama](#deployment-with-ollama)

---

## Overview

The goal is to fine-tune an LLM that can:
- **Present original coding problems** (DSA, System Design, GenAI)
- **Accept candidate solutions** (code submissions)
- **Critique and provide feedback** on incorrect solutions
- **Simulate realistic interview dynamics** with follow-up questions
- **Help improve** by providing hints and step-by-step guidance

---

## Choosing the Right Model

### Recommended Models by Hardware

| VRAM Available | Best Model Choice | Why? |
|----------------|-------------------|------|
| < 8GB | **Qwen 2.5 Coder 1.5B** or **Llama 3.2 3B** | Lightweight, efficient for small GPUs |
| 8-12GB | **Qwen 2.5 Coder 7B** or **Llama 3.1 8B** | Best balance of performance & efficiency |
| 16-24GB | **Mistral-Nemo 12B** or **DeepSeek Coder 6.7B** | Larger models for better reasoning |

### Model Type Selection

**For Pure Coding (LeetCode, DSA):**
- **Best:** Qwen 2.5 Coder 7B
- **Why:** State-of-the-art on HumanEval/MBPP benchmarks, pre-trained on massive code datasets

**For Explanations + Coding:**
- **Best:** Llama 3.1 8B Instruct
- **Why:** Better at natural language explanations while maintaining strong coding ability

**Base vs Instruct Models:**
- **Base Model** (e.g., `Llama-3.1-8B`): Raw pretrained model, harder to control
- **Instruct Model** (e.g., `Llama-3.1-8B-Instruct`): Already fine-tuned for chat/instructions
- **Recommendation:** Start with **Instruct** versions—they converge faster and understand instruction formats better

---

## Environment Setup

### Prerequisites
- **Windows**: Use WSL2 (Windows Subsystem for Linux) for GPU support
- **Alternative**: Google Colab (free GPU) for training, then download model to local machine
- **Storage**: G:\\Interview_Questions_Dataset (Windows path for datasets)

### Installation (WSL2 or Colab)

```bash
# Install Unsloth (fast fine-tuning framework)
pip install unsloth

# Install dependencies
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Install datasets library
pip install datasets
```

### Why Unsloth?
- **2-5x faster** training than standard methods
- **60% less VRAM** usage through optimizations
- **Supports** Llama, Qwen, Mistral out-of-the-box
- **GGUF export** for direct Ollama deployment

---

## Data Preparation

### Dataset Requirements

For an interview simulator, you need data in this format:

```python
{
    "query": "Problem description + starter code",
    "response": "Complete solution with explanation"
}
```

### Using LeetCode Dataset

The `newfacade/LeetCodeDataset` already has this schema:
- **`query`**: Contains problem statement
- **`response`**: Contains solution code

### Interview Mode Enhancement

To make the model act as an interviewer (not just solution generator), mix in conversational examples:

```python
interviewer_examples = [
    {
        "instruction": "Act as a FAANG interviewer. Present a medium-difficulty array problem.",
        "input": "Start interview",
        "response": "Let's begin. Here's your problem: Given an array of integers..."
    },
    {
        "instruction": "Review candidate's solution and provide feedback.",
        "input": "[Candidate's code]",
        "response": "Your approach works for most cases, but has O(n²) complexity. Can you optimize to O(n)?"
    }
]
```

---

## Fine-Tuning Process

### Complete Training Script

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# 1. Configuration
max_seq_length = 2048  # LeetCode problems can be long
dtype = None           # Auto-detect
load_in_4bit = True    # Use 4-bit quantization to save memory

# 2. Load Model (Qwen 2.5 Coder - Best for coding)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Alternative: Use Llama 3.1 for better explanations
# model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# 3. Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. Load Dataset
dataset = load_dataset("newfacade/LeetCodeDataset", split="train")

# 5. Format Data with Interview Prompt Template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert coding interviewer. Solve the following problem efficiently and explain your approach.

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    inputs = examples["query"]      # Problem descriptions
    outputs = examples["response"]  # Solutions
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        text = alpaca_prompt.format(input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# 6. Training Configuration
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 300,  # Increase for full training (500-1000)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",  # Use "wandb" for tracking
    ),
)

# 7. Train!
trainer.train()

# 8. Save Model (GGUF format for Ollama)
save_path = "G:/Interview_Questions_Dataset/LeetCode_Interviewer_Model"
model.save_pretrained_gguf(
    save_path, 
    tokenizer, 
    quantization_method = "q4_k_m"  # 4-bit quantization
)

print(f"Model saved to: {save_path}")
```

### Training Parameters Explained

- **`per_device_train_batch_size = 2`**: Process 2 samples at once (adjust based on VRAM)
- **`gradient_accumulation_steps = 4`**: Effective batch size = 2 × 4 = 8
- **`max_steps = 300`**: Number of training iterations (increase to 500-1000 for production)
- **`learning_rate = 2e-4`**: Standard for fine-tuning
- **`q4_k_m` quantization**: 4-bit for smaller file size, works great with Ollama

---

## Creating an Interview Simulator

### System Prompt for Interview Mode

When running the fine-tuned model, use this system prompt:

```python
SYSTEM_PROMPT = """
You are a senior software engineer conducting technical interviews for FAANG companies.

Your responsibilities:
1. Present coding problems appropriate to the candidate's level
2. Observe the candidate's approach and problem-solving process
3. Provide constructive feedback on their solutions
4. Point out bugs, inefficiencies, or edge cases they missed
5. Guide them toward better solutions with hints (don't give away answers immediately)
6. Ask follow-up questions about time/space complexity
7. Simulate realistic interview pressure and dynamics

Be professional, encouraging, but maintain high standards.
"""
```

### Multi-Turn Conversation Flow

```python
# Example interview session
conversation = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "I'm ready to start the interview."},
    {"role": "assistant", "content": "Great! Let's begin with a medium difficulty problem..."},
    {"role": "user", "content": "[Your solution code]"},
    {"role": "assistant", "content": "Your solution works, but I notice the time complexity is O(n²)..."},
]
```

### Interactive Prompt Template

For Ollama, create a custom prompt that enables back-and-forth:

```python
interview_prompt = """
You are conducting a coding interview. The candidate has submitted this solution:

```python
{candidate_code}
```

Provide feedback:
1. Does it solve the problem correctly?
2. What's the time/space complexity?
3. Are there edge cases it doesn't handle?
4. How can it be improved?

Be constructive and guide them to a better solution.
"""
```

---

## Testing & Benchmarks

### Industry-Standard Coding Benchmarks

| Benchmark | Description | Metric | Link |
|-----------|-------------|--------|------|
| **HumanEval** | 164 Python programming problems | pass@k (correctness) | [GitHub](https://github.com/openai/human-eval) |
| **MBPP** | 974 basic Python problems | pass@k | [GitHub](https://github.com/google-research/google-research/tree/master/mbpp) |
| **LiveCodeBench** | Recent LeetCode/Codeforces problems | Generalization | [Website](https://livecodebench.github.io/) |
| **HumanEval Pro** | Multi-turn problem solving | Multi-step reasoning | [Paper](https://arxiv.org/abs/2406.14497) |

### Running Benchmark Tests

```bash
# Install evaluation framework
pip install human-eval

# Run HumanEval
from human_eval.evaluation import evaluate_functional_correctness

# Generate completions with your model
evaluate_functional_correctness(
    sample_file="samples.jsonl",
    k=[1, 10, 100],
    problem_file="HumanEval.jsonl"
)
```

### Custom Interview Simulation Test

Create a test suite to evaluate interviewer capabilities:

```python
interviewer_tests = [
    {
        "scenario": "Present a medium array problem",
        "expected": "Model provides clear problem statement with examples"
    },
    {
        "scenario": "Candidate submits buggy code",
        "expected": "Model identifies bugs and explains them"
    },
    {
        "scenario": "Solution works but is inefficient",
        "expected": "Model suggests optimization strategies"
    },
    {
        "scenario": "Candidate asks for hints",
        "expected": "Model provides gradual hints without revealing full solution"
    }
]
```

### Evaluation Criteria

**For Interview Simulator Quality:**
1. **Problem Clarity** (1-5): Are problems well-explained?
2. **Feedback Quality** (1-5): Is critique constructive and accurate?
3. **Hint Effectiveness** (1-5): Do hints guide without giving away answers?
4. **Realism** (1-5): Does it feel like a real interview?
5. **Adaptability** (1-5): Does it adjust difficulty based on performance?

---

## Deployment with Ollama

### Step 1: Export Model to GGUF

After training, your model is saved in GGUF format:
```
G:/Interview_Questions_Dataset/LeetCode_Interviewer_Model/
└── LeetCode_Interviewer_Model-unsloth.Q4_K_M.gguf
```

### Step 2: Create Ollama Modelfile

Create a file named `Modelfile` in the same directory:

```dockerfile
# Modelfile
FROM ./LeetCode_Interviewer_Model-unsloth.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""

SYSTEM """
You are a senior software engineer conducting coding interviews for top tech companies. 
Present problems, evaluate solutions, provide constructive feedback, and guide candidates to better approaches.
Maintain professional interview dynamics while being encouraging.
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER stop "<|end|>"
PARAMETER stop "<|eot_id|>"
```

### Step 3: Import into Ollama

```powershell
# Navigate to model directory
cd G:\Interview_Questions_Dataset\LeetCode_Interviewer_Model

# Create the model in Ollama
ollama create coding-interviewer -f Modelfile

# Verify creation
ollama list
```

### Step 4: Run Your Interviewer

```powershell
# Start interactive interview session
ollama run coding-interviewer
```

**Example Interaction:**
```
>>> I'm ready for the interview

Great! Let's start with a medium-difficulty problem. 

Problem: Given an array of integers, find two numbers that add up to a specific target.

Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9

Take your time to think through the approach. Let me know when you're ready to code.

>>> [You submit your solution]

Good effort! Your brute force approach works, but let's discuss the time complexity...
```

### Step 5: Integration with Applications

```python
# Python script to interact with Ollama model
import requests
import json

def interview_chat(prompt, model="coding-interviewer"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# Start interview
print(interview_chat("Give me a hard dynamic programming problem"))
```

---

## Advanced Techniques

### Multi-Dataset Training

Combine multiple datasets for better coverage:

```python
from datasets import load_dataset, concatenate_datasets

# LeetCode problems
leetcode_data = load_dataset("newfacade/LeetCodeDataset", split="train")

# System design questions (if available)
system_design_data = load_dataset("your-org/system-design-qa", split="train")

# Combine
combined = concatenate_datasets([leetcode_data, system_design_data])
```

### Adding Behavioral Interview Capability

Extend beyond coding:

```python
behavioral_examples = [
    {
        "query": "Tell me about a time you faced a technical challenge",
        "response": "That's a great behavioral question. I'll evaluate: (1) Situation clarity, (2) Actions taken, (3) Results achieved, (4) Lessons learned..."
    }
]
```

### Fine-Tuning for Specific Companies

Create company-specific interview datasets:
- **Google**: Focus on algorithms, scale, design
- **Meta**: System design, user impact
- **Amazon**: Leadership principles integration

---

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training
- **Solution**: Reduce `per_device_train_batch_size` to 1
- Use `gradient_accumulation_steps` to maintain effective batch size

**Issue**: Model gives away solutions too easily
- **Solution**: Add more examples where model provides hints instead of full solutions

**Issue**: Model doesn't adapt to candidate level
- **Solution**: Include difficulty progression examples in training data

**Issue**: Ollama model not responding correctly
- **Solution**: Check TEMPLATE format in Modelfile matches your tokenizer's chat format

---

## Resources

### Datasets
- [LeetCodeDataset](https://github.com/newfacade/LeetCodeDataset) - 2000+ LeetCode problems
- [HumanEval](https://github.com/openai/human-eval) - OpenAI coding benchmark
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) - Basic Python problems

### Training Frameworks
- [Unsloth](https://github.com/unslothai/unsloth) - Fast fine-tuning
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - Alternative framework
- [TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning

### Model Hosting
- [Ollama](https://ollama.ai/) - Local model deployment
- [vLLM](https://github.com/vllm-project/vllm) - High-performance serving
- [LM Studio](https://lmstudio.ai/) - GUI for local models

### Benchmarking
- [LiveCodeBench](https://livecodebench.github.io/) - Latest coding problems
- [BigCodeBench](https://github.com/bigcode-project/bigcodebench) - Code generation

---

## Next Steps

1. **Start Small**: Fine-tune Qwen 2.5 Coder 1.5B first to validate the process
2. **Iterate**: Add more conversational examples for interview realism
3. **Benchmark**: Test on HumanEval to measure coding capability
4. **Deploy**: Use Ollama for local testing
5. **Scale**: Move to 7B/8B models for production quality
6. **Collect Feedback**: Use your own interview practice to improve the model

---

## Contributing

Improvements to this guide are welcome! Focus areas:
- Additional company-specific interview datasets
- Behavioral interview integration
- System design problem formats
- Multi-language support (Java, C++, JavaScript)

---

## License

This guide is provided under MIT License. The underlying models and datasets have their own licenses - check before commercial use.

---

**Built for SDE-2 Interview Prep | Target: 25-35 LPA | Mumbai Tech Scene**
