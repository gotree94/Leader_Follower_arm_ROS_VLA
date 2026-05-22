"""
Cosmos-Policy LoRA Fine-tuning
"""
import torch
from vla_policy.vla_network import VLAConfig


def setup_cosmos_policy_lora(base_model_path="nvidia/cosmos-predict2-2b", lora_rank=16, lora_alpha=32):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("peft not installed. Install with: pip install peft")
        return None

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def finetune_cosmos_policy(model, train_dataset, val_dataset, output_dir="models/cosmos_policy_lora"):
    from transformers import TrainingArguments, Trainer
    args = TrainingArguments(
        output_dir=output_dir, per_device_train_batch_size=4,
        gradient_accumulation_steps=8, num_train_epochs=10,
        learning_rate=1e-4, fp16=True, save_steps=500,
        logging_steps=10, evaluation_strategy="steps", eval_steps=500)
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset)
    trainer.train()
    model.save_pretrained(f"{output_dir}/lora_adapter")
    print(f"LoRA adapter saved to {output_dir}/lora_adapter")
    return model
