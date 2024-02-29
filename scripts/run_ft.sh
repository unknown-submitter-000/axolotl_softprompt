pip install transformers==4.36.1
dataset_name=pubmed;
pos_type=textual;
encoder_type=angle;
model="vicuna-7b-v1.5";
relevance_type="pos";
bittype="8bit";
epochs=15;
mkdir ${dataset_name}
cd ${dataset_name}
mkdir ${relevance_type}
cd ${relevance_type}
for order in 2 0;
do
mkdir ${pos_type}_order_${order}_${encoder_type}
cd ${pos_type}_order_${order}_${encoder_type}
cat << EOF > "${model}_${bittype}.yml"
base_model: /home/ubuntu/proj/llm_models/${model}
base_model_config: /home/ubuntu/proj/llm_models/${model}
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
is_llama_derived_model: true

load_in_8bit: true
load_in_4bit: false
strict: false

datasets:
  - path: /home/ubuntu/proj/code/axolotl_softprompt/data/${dataset_name}/train.jsonl
    has_soft: true
    pos_path: /home/ubuntu/proj/code/axolotl_softprompt/data/${dataset_name}/${relevance_type}/train_${pos_type}_order${order}-${encoder_type}.pt
    type: alpaca
dataset_prepared_path:
val_set_size: 0.05
output_dir: ./${model}_${bittype}_${dataset_name}_${pos_type}_order_${order}_epochs_${epochs}_${encoder_type}

adapter: softprompt
adapter_model_dir:

sequence_len: 256
sample_packing: false
pad_to_sequence_len: true

softprompt_input_embedding_dim: 1024
softprompt_num_pos_tokens: 1
softprompt_num_virtual_tokens: 4
softprompt_encoder_hidden_size: 1024

wandb_project:
wandb_entity:
wandb_watch:
wandb_run_id:
wandb_log_model:

gradient_accumulation_steps: 1
micro_batch_size: 16
num_epochs: ${epochs}
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: true
fp16: false
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
eval_steps: 0.05
eval_table_size:
save_steps: 500
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  bos_token: "<s>"
  eos_token: "</s>"
  unk_token: "<unk>"
EOF
accelerate launch -m axolotl.cli.train ${model}_${bittype}.yml >& ${model}_${bittype}_${dataset_name}_${pos_type}_${order}_${encoder_type}_epochs${epochs}.out
cd ..
done