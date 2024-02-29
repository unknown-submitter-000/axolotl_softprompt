pip install transformers==4.35.1
dataset_name=pubmed;
pos_type=textual;
model="vicuna-7b-v1.5";
bittype="8bit";
relevance_type="pos";
encoder_type=angle;
epochs=15;
for order in 0;
do
accelerate launch -m softprompt.evaluate.graph_inference-soft-ds --task node \
                                                                 --dataset-name ${dataset_name} \
                                                                 --input-path /home/ubuntu/proj/code/axolotl_softprompt/data \
                                                                 --pos-name ${pos_type}_order${order}-${encoder_type} \
                                                                 --model-path /home/ubuntu/proj/llm_models \
                                                                 --model vicuna-7b-v1.5 \
                                                                 --adapter-path /home/ubuntu/proj/code/axolotl_softprompt/scripts/${dataset_name}/${relevance_type}/${pos_type}_order_${order}_${encoder_type}/${model}_${bittype}_${dataset_name}_${pos_type}_order_${order}_epochs_${epochs}_${encoder_type} \
                                                                 --batch-size 16 \
                                                                 --encoder-type ${encoder_type} \
                                                                 --relevance-type ${relevance_type} \
                                                                 --output-path /home/ubuntu/proj/code/axolotl_softprompt/scripts/${dataset_name}/${relevance_type}/${pos_type}_order_${order}_${encoder_type}/${model}_${bittype}_${dataset_name}_${pos_type}_order_${order}_epochs_${epochs}_${encoder_type}                                       
done
