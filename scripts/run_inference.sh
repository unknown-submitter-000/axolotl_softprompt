dataset_name=pubmed;
pos_type=textual;
model="vicuna-7b-v1.5";
bittype="8bit";
for order in 2;
do
accelerate launch -m softprompt.evaluate.graph_inference-soft-ds --task node \
                                                                 --dataset-name ${dataset_name} \
                                                                 --input-path /home/ubuntu/proj/code/axolotl_softprompt/data \
                                                                 --pos-name ${pos_type}_order${order} \
                                                                 --model-path /home/ubuntu/proj/llm_models \
                                                                 --model vicuna-7b-v1.5 \
                                                                 --adapter-path /home/ubuntu/proj/code/axolotl_softprompt/scripts/${dataset_name}/${pos_type}_order_${order}/${model}_${bittype}_${dataset_name}_${pos_type}_order_${order} \
                                                                 --batch-size 16 \
                                                                 --output-path /home/ubuntu/proj/code/axolotl_softprompt/scripts/${dataset_name}/${pos_type}_order_${order}/${model}_${bittype}_${dataset_name}_${pos_type}_order_${order}                                           
done
