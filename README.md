# Introduction
Basal cell carcinoma (BCC) is the most common cancer in humans. The high incidence rates cause a significant burden at pathology laboratories. The standard process of identifying tumors is through the analysis of histological images, which is a time-consuming process and prone to inter-pathologist variability. To mitigate these challenges, different deep learning approaches have been applied in many cancer subtype classification tasks. However, there is limited literature on the application of vision transformers to BCC on whole slide images (WSIs). Moreover, there is a lack of BCC dataset representing different aggressivity grades annotated on WSI-level. 

Towards addressing the gap, we scanned 1832 BCC whole slide images (WSIs) obtained from 479 patients. The WSIs weakly annotated into four subtypes of which two (superficial and nodular) represented low aggressive tumors, and two (medium and high risk) representing variants of aggressive BCC subtypes. Then, we used combination graph neural networks and vision transformers with multi-scale feature fusion of WSIs to 1) detect the presence of tumor (two classes), 2) classify the tumor into low and high-risk subtypes (three classes including no tumor), and 3) classify four subtypes (five classes including no tumor).

Using the approach a mean accuracy 89.4 ± 2.0 %, 82.2 ± 1.9% and 68.8 ± 1.6% was obtained on test set, based on models of 5-fold cross-validation of two, three, and five classes classifications, respectively.  

These results show close to state-of-the-art in accuracy in both tumor detection and grading of BCCs. The use of automated image analysis could increase workflow efficiency and even possibly overcome the inter-pathologist variability when grading BCCs. 

# Time stamps
Many attempts have been made to improve the method and get better results. Few of the attempts can be logged in the file `time_stamp.txt`. 

# Dataset

The dataset is stored in `P:\AI.Repository\BCC projekt-digital`. To get access, contact AI-platform support.
On the AI platform, the slides are stored in `/data/BCC projekt-digital`. You can install [qupath](https://qupath.github.io/) in your local machine to visualize slides.

# Annonymize slides (optional)
The slides were scanned using a scanner NanoZoomer S360 Hamamatsu at 40X magnification and saved as .ndpi files. The tool used for anonymizing the data set is found [here](https://github.com/bgilbert/anonymize-slide). According to the developer, the "Slide files are modified in place, making this program both fast and potentially destructive. Do not run it on your only copy of a slide." You can see the details of the implementation in `do_hamamatsu_ndpi(filename)` in `anonymize-slide.py`. The underlying steps appear to be removing the [macro images](https://en.wikipedia.org/wiki/Image_macro) (slide label) at the side of the main images. Since the tool uses python 2.7, it is convenient to create a virtual environment and run the anonymization process following these steps:

```
git clone https://github.com/bgilbert/anonymize-slide.git
conda create -n anonym_ndpi python=2.7
conda activate anonym_ndpi
cd to/path/copy/of/orginal/slides                       # recommended to anonymize copied slides
python H:/anonymize-slide/anonymize-slide.py *.ndpi
```
Note: The dataset used in this project is already anonymized!

# Login to the AI-platform and submit a docker image

Log in to runai following steps [here](https://git.vgregion.se/aiplattform/docs/researchers/-/blob/main/docs/runai_get_started.md) 

The docker image for this project is available at `registry.git.vgregion.se/ai-researchers/bcc`

submit the job using

`runai submit bcc -i registry.git.vgregion.se/ai-researchers/bcc:latest --image-pull-policy Always -g 2 --host-ipc host --service-type nodeport --port 30900:8888 --port 30901:8080 --run-as-user --interactive --preemptible --pvc bcc-pathologi-pvc:/workspace/data -- bash -c "\"export HOME=/workspace; jupyter lab --ServerApp.token=******** --port=8888 --no-browser --ip=0.0.0.0 --allow-root\""`

*edit request: can we get a submit string that follows the current standard? (found at https://git.vgregion.se/aiplattform/docs/researchers/-/blob/main/docs/runai_get_started.md)* 

# Copy slides from P: drive to AI-platform

Contact AI-platform support to copy slides to the AI platform, if there is a need.

To see slide stats, check out `AICC/split_train_val_patient.ipynb`

# Classification using double magnification of WSIs
## Method
  ![GT method](/figures/method figure.png)

## Move, Tile, and remove slides 
To save disk space the .ndpi slides are moved to a temporary folder `bcc`, tiled and removed every minute. Run the scripts below into two separate terminals. 

- `python move_and_tile_ndpis.py` 
- `python remove_ndpi.py` 

The script `move_and_tile_ndpis.py` calls `deepzoom_tiler.py -m 2 3 -b 40 -v ndpi -j 32 --dataset bcc`. The default output patch size is (244, 244) and should match to that in `simclr/config.yaml` (see below). To tile at different magnifications change `-m 2 3` to the desired magnification levels. For example, `-m 1 2 -b 40` generates magnification at 20X and 10X (level 0, level 1, level 2, and level 3 correspond to 40X, 20X, 10X, and 5X, respectively). To see examples of different levels, use the function `show_level_dims()` in `AICC/histolab_grid`. The base magnification for the dataset is 40X. See also [OpenSlide](https://openslide.org/api/python/) for details. The tiled slides will have this structure

```
root
|-- WSI
|   |-- bcc
|   |   |-- pyramid
|   |   |   |-- data
|   |   |   |   |-- BCC (x)_y_JS_z
|   |   |   |   |   |-- PATCH_LOW_1
|   |   |   |   |   |   |-- PATCH_HIGH_1.jpeg
|   |   |   |   |   |   |-- PATCH_HIGH_2.jpeg
|   |   |   |   |   |-- ...
|   |   |   |   |   |-- PATCH_LOW_1.jpeg
|   |   |   |   |   |-- ...
```

Note: This project used 5X and 10X magnifications _only_. The higher the magnification level the largest the number of patches generated, hence the longer training time. The optimal magnification is the one that is low enough to capture the context and high enough to capture the morphology of tumor cells, at the same time, compromise in small gain in accuracy vs training time. When the two magnifications are fused, the number of patches will be the same as low magnification, hence, can be trained in a relatively shorter time. 

*edit request: can move_and_tile_ndpis.py and python remove_ndpi.py be combined to a single workflow?*

[Separation_of_concerns](https://en.wikipedia.org/wiki/Separation_of_concerns) 

## Install requirements
While copying files, which takes upto 3-4 days to complete, start installing the requirements using 

```
pip install -r requirements_gt.txt # will take up to 2 hours to install dependencies
```
Note: In the current configuration of AI platform and BCC docker image, adding the above requirements to the requirements list of the image creates dependency conflicts in downstream. It is recommended that you install the requirements separately.

## Self-supervised learning layer (SimCLR) 

Make changes to `simclr/config.yaml` file as necessary. The [SimCLR](https://arxiv.org/pdf/2002.05709.pdf) paper shows the benefits of large batch size and epochs. In this project, 512 batch size and 32 epochs are used (However, the training can continue to get better results. Use fine tune option in the `simclr/config.yaml` file). For each magnification (5X and 10X) train SimCLR separately. To start training, run

```
cd simclr 
python run.py --multiscale=1 --level=low   --dataset bcc --wsi_address /workspace/data/cv_methods/tmi2022/
python run.py --multiscale=1 --level=high  --dataset bcc --wsi_address /workspace/data/cv_methods/tmi2022/
```
In the current implementation, the code is pointing to the `WSI` and `WSI_test_set` folders in `/workspace/data/cv_methods/tmi2022/`. There is a copy of the dataset in `/workspace/data/bcc_data_and_models`. Either of this folders can removed depending on future plans. All the steps in document, however, should point to either of this two folders. 

You can see training progress using

```
cd simclr/runs
tensorboard --logdir Oct05_21-53-25_bcc-0-0/ --host 0.0.0.0 --port 8080  
```

[http://vgas3925.vgregion.se:yourportmap/](http://vgas3925.vgregion.se:30901/). You can get the "port map", the one you used when submitting a job. In this example, 30901)

Note: the current training used only 32 epochs.

## Feature extraction 
To extract embeddings from the trained SimCLR (mainly from CNN-layer), set the right path of the weights of the model using `--weights_low` and `--weights_high` tags. This step will fuse both embeddings from the two magnifications and save them in a folder `datasets`. By default, the fusion is performed by averaging the high-magnification embeddings and adding them to the weighted embeddings of low magnification (see the paper for more details). The embeddings can be concatenated by setting the parameter `fusion` to `cat` in the function call `compute_tree_feats(..., fusion='cat')`.  (In to-do list: change the dates below to '_low' and '_high' programmatically)

`python compute_feats.py --dataset bcc --batch_size 8 --num_classes 5 --magnification tree --weights_low /workspace/data/cv_methods/tmi2022/simclr/runs/low_mag_Oct05_21-53-25_bcc-0-0 --weights_high /workspace/data/cv_methods/tmi2022/simclr/runs/high_mag_Oct07_10-36-46_bcc-0-0`

Note: The low magnification images are likely to have low contribution to relative to high magnification images. Hence, weighted average of the two magnification is used. The weighted average can be modified by changing the values` _weight_low = 0.25 `in `compute_feats.py`. See the feature extraction section of the paper for more details.

## Generate graphs 
Now that the embeddings are generated, a representative graph can be generated for each slide. The main idea is that each feature vector of each patch is a node on the graph and the connection is the spatial correlation among the patches.

```
cd simclr
python build_graphs_double.py --weights  /workspace/data/cv_methods/tmi2022/simclr/runs/low_mag_Oct05_21-53-25_bcc-0-0/checkpoints/model.pth --dataset "/workspace/data/cv_methods/tmi2022/WSI/bcc/pyramid/data/*/" --output "../graphs" --num_classes 5
```
Note: The graphs are build from the fused dataset. Hence, `--weights  /workspace/data/cv_methods/tmi2022/simclr/runs/low_mag_Oct05_21-53-25_bcc-0-0/checkpoints/model.pth` is bypassed in the code. Again,`--dataset /workspace/data/cv_methods/tmi2022/WSI/bcc/pyramid/data/*/` is also bypassed in the code. To-do list: remove the bypass lines in the code.

## Generate train and validation sets 
The section `Split train, validation and test` in the notebook `AICC/split_train_val_patient.ipynb` is used to generate a 5 Fold cross-validation dataset for 2, 3, and 5 classes based on extraction ID (not patient ID). The 2 and 3 classes datasets are generated from 5 classes dataset. (However, there is an option to generate each class independently). The dataset is saved in the form of `script/train_set_3C_1F.txt` for the training set and `scripts/val_set_3C_1F.txt` for the validation set. The C refers to number of classes and the F refers to fold number. (To-do list: seed kfold generator.)

## Training graph-transformer on double magnification

Since there are 13 files for training, validation, and test for each of the 3 tasks (30+ files, without including single magnification), it is safer to automate the configurations for each fold and task. The configurations can be generated for each task in the form of `scripts/train_cv_double_x.sh`. Change the parent paths in the template config file`train_double.sh` to your parent path. Make sure the variable `simclr_files_folder` is assigned to `'simclr_files_double'` in `utils/dataset.py`. To generate config files for each fold and task, using

 `AICC/generate_cross_val_fold_scripts.ipynb ` 

All these files can then be run one at a time using

`bash scripts/train_all_double.sh`

Note:

(1) The number of GPUs set by `CUDA_VISIBLE_DEVICES=0,1`in `scripts/train_cv_double_x.sh` should match the available GPUs. Currently, the desire of GPUs is set by `model = nn.DataParallel(model, device_ids=[0,1])` in `main.py`. Parallel training works properly one model at a time. When using `bash scripts/train_all_double.sh` that trains multiple models use only one GPU.

(2) You can see training progress, ROC curve and confusion matrix in `AICC/multi_class_metrics.ipynb`

(3) The hyperparameters number of epochs and learning rate are located in `options.py` and batch_size in `scripts/train_cv_double_x.sh`. The training scheduler and adam's weighted decay are located in` main.py `. 

# Predict using trained model on test set
The slides of test set are saved in `bcc-project-digial/test set`. Move the data and tile it using 
```
cd bcc-projekt-digital
mv test\ set/* bcc/
cd /workspace/data/cv_methods/tmi2022
python deepzoom_tiler.py -m 2 3 -b 40 -v ndpi -j 32 --dataset bcc --holdout_set True
```
The tiled files will be saved in `WSI_test_set`

To extract features vectors and save them in `datasets_test_set` folder, use

`python compute_feats.py --dataset bcc --batch_size 8 --num_classes 5 --magnification tree --weights_low /workspace//data/cv_methods/tmi2022/simclr/runs/low_mag_Oct05_21-53-25_bcc-0-0 --weights_high /workspace/data/cv_methods/tmi2022/simclr/runs/high_mag_Oct07_10-36-46_bcc-0-0 --holdout_set True`

To build graphs and save them in `graphs/simclr_files_double`, use

```
cd simclr
python build_graphs_double.py --weights  /workspace/data/cv_methods/tmi2022/simclr/runs/low_mag_Oct05_21-53-25_bcc-0-0/checkpoints/model.pth --dataset "/workspace/data/cv_methods/tmi2022/WSI/bcc/pyramid/data/*/" --output "../graphs" --num_classes 5 --holdout_set True
```

To infer using the test set, first generate the list of test slides using the function `create_test_set_list()` in `split_train_val_patient.ipynb`and the `function write_cross_test(fold=5, clas=2)` in `generate_cross_val_fold_scripts.ipynb` (Change the parent paths in the template config file`test.sh` your parent path). This will generate the necessary configuration for testing for a specific task (classes 2, 3, or 5. Change the variable `class` to a desired class). Then run 

`bash scripts/test_all_double.sh `

You will see the results in `val_test_results` folder.

## Explainability/ interpretability/ visualization
To perform interpretability of model or visualize the result the following steps. 

- Save the filename of the slide you want to visualize in `scripts/test_set_GT_build_cam.txt` in the form of `BCC550 (1)` followed by 0 (as in tab separated).
- Pick the task type (e.g. 2 class, 3 class, 5 class classification)
- Open `scripts/get_graphcam.sh` and `scripts/vis_graphcam.sh` change the configuration according to the task at hand. Change `--resume ` path to a trained model and the patch folder `"/workspace/data/cv_methods/tmi2022/WSI_test_set/bcc/pyramid/data/BCC332 (4)"` \
- Open `models/GraphTransofermer.py` and change `num_classes` to the desired class. 
- Make sure the variable `simclr_files_double` in `utils/datasets.py` is set to right `graphs` folder.
- To generate the class activation mapping and class probabilities run

`bash scripts/get_vis_graphcam_AICC.sh`

Running the above step will run two `.sh` scripts. Make sure the paths are set properly.
1. `bash scripts/get_graphcam.sh`   # Choose only one GPU (e.g. `CUDA_VISIBLE_DEVICES=0`) as it does not work on parallel GPUs
2. `bash scripts/vis_graphcam.sh`   # rename the file name

The above steps will generate colored masks overlaying the slide image. This uses Class Activation Mapping, Layerwise Relevance Propagation which compute "relevancies" backward to the input image. The results are saved in the folder `graphcam_vis`. For better visualization change `p = np.clip(p, 0.8, 1)` to desired range in `src/vis_graphcam_AICC.py`.
 
  ![Heatmap](/figures/BCC20_6_1a_bcc_cam_all.png)

# Single magnification
### Option 1: Feature extraction considering high magnification images independently

(1) For training set:

`python compute_feats.py --dataset bcc --batch_size 8 --num_classes 5 --magnification low --weights_low /workspace/data/cv_methods/tmi2022/simclr/runs/low-low_mag_Oct05_21`

`python compute_feats.py --dataset bcc --batch_size 8 --num_classes 5 --magnification high --weights_high /workspace/data/cv_methods/tmi2022/simclr/runs/high_mag_Oct05_21 `

(2) For test set:

`python compute_feats.py --dataset bcc --batch_size 8 --num_classes 5 --magnification low --weights_low /workspace/data/cv_methods/tmi2022/simclr/runs/low-low_mag_Oct05_21 --holdout_set True`

`python compute_feats.py --dataset bcc --batch_size 8 --num_classes 5 --magnification high --weights_high /workspace/data/cv_methods/tmi2022/simclr/runs/high_mag_Oct05_21 --holdout_set True`

Note: these results are saved in `datasets` and `datasets_test_set`

### Option 2: Single magnification tiled to one magnification

Tile to a single magnification, say 10X (m=2), train SimCLR, and generate feature vector, use

```
Python deepzoom_tiler.py -m 2 -b 40 -v ndpi -j 32 --dataset bcc
python run.py  --dataset bcc
python compute_feats.py --dataset bcc --batch_size 8 --num_classes 5 --magnification low --weights /workspace/data/cv_methods/tmi2022/simclr/runs/Oct05_21
```

## build graphs
(1) Training low:
`python build_graphs_double.py --weights  /workspace/data/cv_methods/tmi2022/simclr/runs/low_mag_Oct05_21-53-25_bcc-0-0/checkpoints/model.pth --dataset "/workspace/data/cv_methods/tmi2022/WSI/bcc/pyramid/data/*/" --output "../graphs" --num_classes 5  --separate_feat 'low'`

(2) Training high:
`python build_graphs_double.py --weights  /workspace/data/cv_methods/tmi2022/simclr/runs/low_mag_Oct05_21-53-25_bcc-0-0/checkpoints/model.pth --dataset "/workspace/data/cv_methods/tmi2022/WSI/bcc/pyramid/data/*/" --output "../graphs" --num_classes 5  --separate_feat 'high'`

(3) Test low:
`python build_graphs_double.py --weights  /workspace/data/cv_methods/tmi2022/simclr/runs/low_mag_Oct05_21-53-25_bcc-0-0/checkpoints/model.pth --dataset "/workspace/data/cv_methods/tmi2022/WSI/bcc/pyramid/data/*/" --output "../graphs" --num_classes 5 --holdout_set True  --separate_feat 'low'`

(4) Test high:
`python build_graphs_double.py --weights  /workspace/data/cv_methods/tmi2022/simclr/runs/low_mag_Oct05_21-53-25_bcc-0-0/checkpoints/model.pth --dataset "/workspace/data/cv_methods/tmi2022/WSI/bcc/pyramid/data/*/" --output "../graphs" --num_classes 5 --holdout_set True  --separate_feat 'high'`

Note: 
(1) `--weights  /workspace/data/cv_methods/tmi2022/simclr/runs/low_mag_Oct05_21-53-25_bcc-0-0/checkpoints/model.pth` will be bypassed.
(2) If you take "Option 2: Single magnification tiled to one magnification" to generate the embeddings, use `--separate_feat 'low'` when constructing graph NN. 

## Generate config files
Run `single magnification section` in `generate_cross_val_fold_scripts.ipynb` with desired parameters for each task. Change the parent path in the template `test_low_high.sh ` to your parent path. 

Assign the variable `simclr_files_folder` to either `'simclr_files_double_low'` or `'simclr_files_double_high'` in `utils/dataset.py`

## Train for single magnification
Run `bash scripts/train_all_low_high.sh `

## Test for single magnification
Run `bash scripts/test_all_low_high.sh `

Note: 
(1) The combination of 3 tasks, 5 folds, and low and high magnifications generated 15 models.

(2) You can see the results in the `val_test_results` folder

# Notebooks in AICC folder

The following notebooks are fairly independent of each other, and thus can run in any order. The specific use of the notebooks is mentioned above. 

- The figures in the paper are generated using the notebook `split_train_val_patient.ipynb` , `multi_class_metrics.ipynb` and `draw_graph.ipynb`.

**Frequently used notebooks**
- `split_train_val_patient.ipynb` To get stat. about patients and the datasets
- `generate_cross_val_fold_scripts.ipynb` To generate configuration files (`.sh`) for task specific training, validation, and test purposes
- `multi_class_metrics.ipynb` To see training progress, ROC curve, confusion matrix, and comparison accuracy results

**Seldom used notebooks**
- `draw_graph.ipynb` To draw graph neural networks
- `missclassified.ipynb` To see the list of slides that the model wrongly classified.
- `ensemble_stalk.ipynb` To ensemble 5 fold cross validation models and get better accuracy 

**Rarely used notebooks**
- `move_files.ipynb` Has functions for moving files to different categories based on their class labels
- `select_images.ipynb` To random generate patches from a given slide and pick the desired patch
- `view_patches.ipynb` To see an example of subplots of a slide
- `histolab_grid.ipynb` To see different magnification levels and random grid on top of slides
- `check_slide_quality.ipynb`  To see pixel size numbers per level of slide. (You have to download the notebook to your local machine first to use it)

# UI 
In the long run, these models will need to be wrapped by some sort of interface before being put into practice. We envisioned a web-based solution and implemented a UI with help of Django. The code is available [here](https://git.vgregion.se/digital_foui/bcc_ui). 

# To-do-list
- Add color bars to the heatmaps
- put `compute_feats.py` and `simclr/build_graphs_double.py` into single `.py` file
- The current results include only feature vector fusion (see Feature extraction section). It would be interesting to check if feature vector concatenation gives better results. 

# References
- [A graph-transformer for whole slide images classification](https://arxiv.org/pdf/2205.09671.pdf),([code](https://github.com/binli123/dsmil-wsi)).

- [Multi-scale Domain-adversarial Multiple-instance CNN for Cancer Subtype
Classification with Unannotated Histopathological images](https://arxiv.org/pdf/2001.01599.pdf),([code](https://github.com/vkola-lab/tmi2022)).


