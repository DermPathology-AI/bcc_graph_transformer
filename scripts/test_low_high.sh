export CUDA_VISIBLE_DEVICES=0,1
python main.py \
--n_class 3 \
--data_path  "/workspace/data/cv_methods/gt/graphs/simclr_files_double_low/" \
--val_set    "/workspace/data/cv_methods/gt/scripts/test_set_2C_4F.txt" \
--model_path "/workspace/data/cv_methods/gt/graph_transformer/saved_models/" \
--log_path   "/workspace/data/cv_methods/gt/graph_transformer/runs/" \
--task_name  "GT_3C_0F_low_test" \
--batch_size 2 \
--test  \
--log_interval_local 1 \
--resume "/workspace/data/cv_methods/gt/graph_transformer/saved_models/GT_3C_0F_low.pth"
