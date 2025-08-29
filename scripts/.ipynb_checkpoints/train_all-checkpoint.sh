printf 'Will train all the 5-folds'
bash /workspace/data/cv_methods/tmi2022/scripts/train_cv_0.sh
printf 'Fold 0 done!'
bash /workspace/data/cv_methods/tmi2022/scripts/train_cv_1.sh
printf 'Fold 1 done!'
bash /workspace/data/cv_methods/tmi2022/scripts/train_cv_2.sh
printf 'Fold 2 done!'
bash /workspace/data/cv_methods/tmi2022/scripts/train_cv_3.sh
printf 'Fold 3 done!'
bash /workspace/data/cv_methods/tmi2022/scripts/train_cv_4.sh
printf 'Fold 4 done!'
printf 'done'