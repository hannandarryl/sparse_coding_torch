# sparse_coding_torch

Need to go to utils/load_data.py and change where bamc data is located at top of each method

Main sparse coding file is found at:
feature_extraction/train_conv3d_sparse_model.py

Sparse coding command example:
python feature_extraction/train_conv3d_sparse_model.py --output_dir feature_extraction/output


Main classification model is found at:
data_classifiers/small_data_classifier.py

Classification command example:
python data_classifiers/small_data_classifier.py --output_dir data_classifiers/output --checkpoint feature_extraction/output/sparse_conv3d_model-best.pt
