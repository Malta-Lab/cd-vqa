export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1 python main.py \
--data_path /C/lucas/datasets/css_vqa/data/ \
--annotations_path /C/lucas/datasets/css_vqa/annotations/ \
--embed_path /C/lucas/datasets/css_vqa/embeddings/ \
--img_path /C/datasets/vqa/2018-04-27_bottom-up-attention_fixed_36/ \
--qvp 5 \
--top_hint 9 \
--seed 0 \
--output cdvqa \
--entropy_penalty 0.1 \
--debias_weight 1 \
--gpu_id 1 \
--batch_size 512 \
--top_n_answ 5
