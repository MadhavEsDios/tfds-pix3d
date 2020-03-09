#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:21:28 2020

@author: Madhav
"""
import tensorflow_datasets.public_api as tfds
import os
import json
import tensorflow as tf
import scipy.io
import numpy as np

class pix3d(tfds.core.GeneratorBasedBuilder):
  """Short description of my dataset."""

  VERSION = tfds.core.Version('0.1.0')
  MANUAL_DOWNLOAD_INSTRUCTIONS = 'Testing'

  def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=("This is the dataset for Pix3D. It contains yyy. The "
                         "images are kept at their original dimensions."),
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                    'image': tfds.features.Image(),
                    'width': tfds.features.Tensor(shape=(1,), dtype=tf.uint8),
                    'height': tfds.features.Tensor(shape=(1,), dtype=tf.uint8),
                    '2d_keypoints': tfds.features.Tensor(shape=(None, None, 2), dtype=tf.float32),
                    'mask': tfds.features.Image(),
                    '3d_keypoints': tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
                    'voxel': tfds.features.Tensor(shape = (None, None, None), dtype=tf.uint8),
                    'rot_mat': tfds.features.Tensor(shape = (3, 3), dtype=tf.float32),
                    'trans_mat': tfds.features.Tensor(shape = (1, 3), dtype=tf.float32),
                    'focal_length': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                    'cam_position': tfds.features.Tensor(shape=(3, ), dtype=tf.float32),
                    'inplane_rotation': tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                    'bbox': tfds.features.BBoxFeature(),
                    'metadata': {
                            'category': tfds.features.Text(),
                            'img_source': tfds.features.Text(),
                            'model': tfds.features.Text(),
                            #'model_raw': tfds.features.Text(),
                            'model_source': tfds.features.Text()
                            #'truncated': tfds.features.Text(),
                            #'occluded': tfds.features.Text(),
                            #'slightly_occluded': tfds.features.Text()
                            }
            }),
       
            homepage="http://pix3d.csail.mit.edu/",
            # Bibtex citation for the dataset
            citation=r"""@article{my-awesome-dataset-2020,
                                  author = {Smith, John},"}""",
        )
    
  def _split_generators(self, dl_manager):
        # Download source data
        extracted_path = dl_manager.manual_dir#.manual_dir("/Users/Madhav/Experiments/meshrcnn/pix3d")
        #dl_manager.download_and_extract("http://pix3d.csail.mit.edu/data/pix3d.zip")
    
        # Specify the splits
        return [
                tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "extracted_path": extracted_path
                },
            )
        ]
    
    
  def _generate_examples(self, extracted_path):
      json_path = os.path.join(extracted_path, "pix3d.json")
      with open(json_path) as pix3d:
          pix3d_info = json.loads(pix3d.read())
      for pix in pix3d_info:
          image_id = pix['img'][4:-4]
          width, height = pix['img_size']
          normalized_bbox = np.asarray(pix['bbox'], dtype=np.float32) / np.array([width, height, width, height])
          ymin, xmin, ymax, xmax = normalized_bbox
          yield image_id, {
                  "image": os.path.join(extracted_path, pix['img']),
                  "width": np.atleast_1d(width).astype(np.uint8),
                  "height": np.atleast_1d(height).astype(np.uint8),
                  "2d_keypoints": np.asarray(pix['2d_keypoints'], dtype=np.float32),
                  "mask": os.path.join(extracted_path, pix['mask']),
                  "3d_keypoints": np.loadtxt(os.path.join(extracted_path, pix['3d_keypoints']), dtype=np.float32),
                  "voxel": scipy.io.loadmat(os.path.join(extracted_path, pix['voxel']))['voxel'],
                  "rot_mat": np.asarray(pix['rot_mat'], dtype=np.float32),
                  "trans_mat": np.asarray(pix['trans_mat'], dtype=np.float32)[np.newaxis],
                  "focal_length": np.atleast_1d(pix['focal_length']).astype(np.float32),
                  "cam_position": np.asarray(pix['cam_position'], dtype=np.float32),
                  "inplane_rotation": np.atleast_1d(pix['inplane_rotation']).astype(np.float32),
                  "bbox": tfds.features.BBox(ymin = ymin, xmin = xmin, ymax = ymax, xmax = xmax),
                  "metadata": {
                          "category": pix['category'],
                          "img_source": pix['img_source'],
                          "model": pix['model'],
                          #"model_raw": pix['model_raw'].tostring(),
                          "model_source": pix['model_source']
                          #"truncated": pix['truncated'],
                          #occluded": pix['occluded'],
                          #"slightly_occluded": pix['slightly_occluded']
                          }
                  }
        

pix3dbuilder = pix3d()
info = pix3dbuilder.info
#print(info)
pix3dbuilder.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir="/Users/Madhav/Experiments/meshrcnn"))
pix3dbuilder.as_dataset()
