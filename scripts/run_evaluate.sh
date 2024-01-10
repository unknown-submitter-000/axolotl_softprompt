dataset_name=pubmed;
pos_type=textual;
model="vicuna-7b-v1.5";
bittype="8bit";
for order in 2;
do
python -m softprompt.evaluate.evaluate_responses --input-path /home/ubuntu/proj/code/axolotl_softprompt/scripts/${dataset_name}/${pos_type}_order_${order}/${model}_${bittype}_${dataset_name}_${pos_type}_order_${order}/output.jsonl \
                                                 --output-path /home/ubuntu/proj/code/axolotl_softprompt/scripts/${dataset_name}/${pos_type}_order_${order}/${model}_${bittype}_${dataset_name}_${pos_type}_order_${order}/output-metrics.jsonl \
                                                 --metric exact                               
done