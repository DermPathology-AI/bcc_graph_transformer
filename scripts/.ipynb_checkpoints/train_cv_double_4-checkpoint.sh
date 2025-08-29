export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class 2 \
--data_path  "/workspace/data/cv_methods/tmi2022/graphs/simclr_files_double/" \
--train_set  "/workspace/data/cv_methods/tmi2022/scripts/train_set_4.txt" \
--val_set    "/workspace/data/cv_methods/tmi2022/scripts/val_set_4.txt" \
--model_path "/workspace/data/cv_methods/tmi2022/graph_transformer/saved_models/" \
--log_path   "/workspace/data/cv_methods/tmi2022/graph_transformer/runs/" \
--task_name  "GT_double_2C_4F" \
--batch_size 4 \
--train  \
--log_interval_local 6 \
