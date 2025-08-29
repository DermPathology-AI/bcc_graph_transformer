export CUDA_VISIBLE_DEVICES=0,1
python main.py \
--n_class 5 \
--data_path  "/workspace/data/cv_methods/tmi2022/graphs/simclr_files_double_low/" \
--train_set  "/workspace/data/cv_methods/tmi2022/scripts/train_set_5C_0F.txt" \
--val_set    "/workspace/data/cv_methods/tmi2022/scripts/val_set_5C_0F.txt" \
--model_path "/workspace/data/cv_methods/tmi2022/graph_transformer/saved_models/" \
--log_path   "/workspace/data/cv_methods/tmi2022/graph_transformer/runs/" \
--task_name  "GT_5C_0F_low" \
--batch_size 4 \
--train  \
--log_interval_local 6 \
