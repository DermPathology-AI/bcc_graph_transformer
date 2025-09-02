export CUDA_VISIBLE_DEVICES=0,1
python main.py \
--n_class 5 \
--data_path  "/graphs/simclr_files_double_high/" \
--train_set  "/scripts/train_set_5C_3F.txt" \
--val_set    "/scripts/val_set_5C_3F.txt" \
--model_path "/graph_transformer/saved_models/" \
--log_path   "/graph_transformer/runs/" \
--task_name  "GT_5C_3F_high" \
--batch_size 4 \
--train  \
--log_interval_local 6 \
