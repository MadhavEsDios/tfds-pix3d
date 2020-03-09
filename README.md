# tfds-pix3d
In this repo, I attempt to create a tensorflow-datasets dataset builder for the Pix3D dataset.

The objective is to extend tensorflow datasets to include the Pix3D dataset and ultimately use it for implementing the MeshRCNN 
paper in tensorflow.

This is as part of the GSoc'20 proposal provided by tensorflow

Work in Progress:

Resolve:
1)  "too many files" error
2) manual_dir option in dl_manager unusable, had to explicitly pass as tfds.download.DownloadConfig
2) boolean metadata

TODO:
1) Write Test scripts
2) Use train-test splits used in MeshRCNN
3) replace Python IO with Gfile
