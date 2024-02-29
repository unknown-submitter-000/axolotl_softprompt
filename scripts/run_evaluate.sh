dataset_name=pubmed;
pos_type=textual;
model="vicuna-7b-v1.5";
bittype="8bit";
relevance_type="pos";
encoder_type=angle;
epochs=15;
for order in 2;
do
python -m softprompt.evaluate.evaluate_responses \
    --input-path /home/ubuntu/proj/code/axolotl_softprompt/scripts/${dataset_name}/${relevance_type}/${pos_type}_order_${order}_${encoder_type}/${model}_${bittype}_${dataset_name}_${pos_type}_order_${order}_epochs_${epochs}_${encoder_type}/output.jsonl \
    --output-path /home/ubuntu/proj/code/axolotl_softprompt/scripts/${dataset_name}/${relevance_type}/${pos_type}_order_${order}_${encoder_type}/${model}_${bittype}_${dataset_name}_${pos_type}_order_${order}_epochs_${epochs}_${encoder_type}/output-metrics.jsonl \
    --metric exact                               
done