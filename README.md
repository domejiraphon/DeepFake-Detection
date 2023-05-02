## This is the code for DeepFake Detection.

To run the code with [self blended](https://github.com/mapooon/SelfBlendedImages)
```
cd anomaly_detector/scripts
sbatch base_sbi.sh
```

To run the code with self blended method and regression head 
```
cd anomaly_detector/scripts
sbatch base_regression.sh
```

To run the code with self blended method, regression head and occlusion augmentation [https://github.com/kennyvoo/face-occlusion-generation]
```
cd anomaly_detector/scripts
sbatch base_occ.sh
```

To run the inference score 
```
cd anomaly_detector/scripts
sbatch array.sh
```

To do inference:
```
cd anomaly_detector
python inference.py --model_dir 8_1/regress_occlusion --use_retinafacenet --batch_size 128 --use_regression --lpips
```

# Anomaly detection important function
- DeepLabV3
This it the model for [segmentation map](https://github.com/VainF/DeepLabV3Plus-Pytorch).
- detection_and_localization This is the experession network
- face_cutout
This is the [face cutout network](https://github.com/sowmen/face-cutout).
- Important file: face_cutout_utils.py
- face_occlusion_generation 
Model to create synthetic occlusion [hands and objects](https://github.com/kennyvoo/face-occlusion-generation).

- face_parsing
Model to create segmentation map for face. [Need to change top deeplabv3](https://github.com/zllrunning/face-parsing.PyTorch).
- Important file: segment_utils.py
Face_Cutout class -> class to segment faces

## Visualizer
Model to visualize a trained neural network ie. [gradcam, eigencam, ...](https://github.com/jacobgil/pytorch-grad-cam)

## dataset_utils.py
This is the file to store all utils that need for dataset.

- [x] GetFace class -> Model to extract face using METCNN

- [x] FaceMesh class -> Model to extract keypoints

- [x] RetinaGetFace -> Model to extract face using retinanet (Prefer this one)

- [x] DeepLabV3_Segment -> Model to segment the face using deeplabv3 

- [x] Occlusion -> Model to add occlusion to the input images

## pipeline.py

- load_data -> to load test and train set

- log_tensorboard -> To log a tensorboard file

- test -> To do inference on test set.

## trainer.py

- Training loop is here

## utils/loss_utils.py

- This is where I implemented the loss.

- The pipeline figure is here, previous_pipeline.eddx

