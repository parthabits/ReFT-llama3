
# LLaMA-3 ReFT Model Fine-Tuning Example

This project demonstrates how to fine-tune the LLaMA-3 8B model using ReFT (Rank-efficient Finetuning) for intervention-based tuning. The code guides us through loading the model, setting up ReFT configuration, and training it on a few-shot dataset.

## Installation

1. Ensuring that we have the required libraries installed. If not, the code will install them for us:
    - `pyreft`: Used for ReFT fine-tuning.
    - `transformers`: HuggingFace's transformer library.
    - `huggingface_hub`: For authentication and loading models from HuggingFace.

    The script automatically installs `pyreft` if it's not available:
    ```python
    try:
        import pyreft
    except ModuleNotFoundError:
        !pip install git+https://github.com/stanfordnlp/pyreft.git
    ```

2. We would also need to log in to your HuggingFace account to access gated models such as LLaMA-3. We can do so using:
    ```python
    from huggingface_hub import notebook_login
    notebook_login()
    ```

## Model Loading

Once logged in, the model is loaded from HuggingFace. We are using `meta-llama/Meta-Llama-3-8B-Instruct`.

```python
model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048,
    padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
```

## ReFT Setup

We set up the ReFT configuration for fine-tuning by specifying the layer and component for intervention. ReFT will apply a low-rank adaptation to these specific parts of the model.

```python
reft_config = pyreft.ReftConfig(representations={
    "layer": 15,
    "component": "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
    low_rank_dimension=4)} )
reft_model = pyreft.get_reft_model(model, reft_config)
```

## Dataset Preparation

Quick adaptation or personalization is performed on a small few-shot dataset related to health queries. We tokenize and prepare the dataset for training.

```python
training_examples = [
    ["What should I do if I have a persistent cough?", "I'm not a medical professional and cannot provide medical advice."],
    ...
]
data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, model,
     [prompt_no_input_template % e[0] for e in training_examples],
    [e[1] for e in training_examples])
```

## Training

The ReFT model is trained using HuggingFace's `Trainer` API. The training configuration is defined as follows:

```python
training_args = transformers.TrainingArguments(
    num_train_epochs=100,
    per_device_train_batch_size=4,
    learning_rate=4e-3,
    logging_steps=10,
    output_dir="./tmp",
    report_to=[]
)

trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model,
    tokenizer=tokenizer,
    args=training_args,
    **data_module)
trainer.train()
```

## Chat with Your ReFT Model

After training, we can interact with our fine-tuned model by providing an instruction. Here's an example where the model is asked about dog breeds:

```python
instruction = "Which dog breed do people think is cuter, poodle or doodle?"
prompt = prompt_no_input_template % instruction
prompt = tokenizer(prompt, return_tensors="pt").to(device)

base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
_, reft_response = reft_model.generate(
    prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True, max_new_tokens=512, do_sample=True,
    eos_token_id=tokenizer.eos_token_id, early_stopping=True
)
print(tokenizer.decode(reft_response[0], skip_special_tokens=True))
```

## Saving and Loading the ReFT Model

After training, the ReFT model can be saved and shared via HuggingFace:

```python
reft_model.set_device("cpu")
reft_model.save(
    save_directory="./reft_to_share",
    save_to_hf_hub=True,
    hf_repo_name="Ronal999/reft_llama3"
)
```

To load the saved ReFT model in the future, we can do:

```python
reft_model = pyreft.ReftModel.load(
    "./reft_to_share", model
)
```

This project shows how easy it is to perform rank-efficient fine-tuning using ReFT and apply it to real-world use cases with minimal data.

