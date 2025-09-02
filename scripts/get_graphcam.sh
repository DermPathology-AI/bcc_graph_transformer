export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class   2 \
--data_path  "/workspace/data/cv_methods/gt/WSI_test_set"  \
--val_set    "/workspace/data/cv_methods/gt/scripts/test_set_GT_build_cam.txt" \
--model_path "/workspace/data/cv_methods/gt/graph_transformer/saved_models/" \
--log_path   "/workspace/data/cv_methods/gt/graph_transformer/runs/" \
--task_name  "GraphCAM_GT_2C_1F" \
--batch_size 4 \
--test \
--log_interval_local 1 \
--resume "/workspace/data/cv_methods/gt/graph_transformer/saved_models/GT_double_2C_1F.pth" \
--graphcam