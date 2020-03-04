import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sagemaker
from sagemaker.tensorflow import TensorFlowPredictor
import requests
import shutil
import json
import numpy as np

class ObjectDetector:


	def __init__(self, predictorEndpointName):
		print('Instantiating ObjectDetector')
		if predictorEndpointName:
			self.predictor = sagemaker.RealTimePredictor(predictorEndpointName)
			self.set_class_labels()
		else:
			print('Must Supply an Endpoint Name')


	def set_class_labels(self):
		print('Adding Class Labels')

		self.object_categories = [ 'person', 'bicycle', 'car',  'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
		'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
		'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
		'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
		'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
		'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable',
		'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
		'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
		'toothbrush']

		print('Added {} Class Labels'.format(len(self.object_categories)))


	def test_model_on_manifest_data(self, bucket_name, manifest):

		image_path = self.generate_random_image_file_from_manifest(bucket_name, manifest)
		self.visualize_detection(image_path)

	def test_model_on_image(self, url_of_image):

		image_path = download_image(self, url, to_save_filename='test-image.png')
		self.visualize_detection(image_path)


	def visualize_detection(self, image_path, thresh=0.6):

		local_path = image_path

		with open(local_path, 'rb') as f:
			img_file = f.read()
			img_file = bytearray(img_file)
			ne = open('n.txt','wb')
			ne.write(img_file)

		classes = self.object_categories
	
		self.predictor.content_type = 'image/jpeg'
		results = self.predictor.predict(img_file)
		dets = json.loads(results)
		dets = dets['prediction']

		img=mpimg.imread(local_path)
		plt.imshow(img)
		height = img.shape[0]
		width = img.shape[1]
		colors = dict()
		det_classes_and_score = {}
		for det in dets:
			(klass, score, x0, y0, x1, y1) = det
			if score < thresh:
				continue
			cls_id = int(klass)
			if cls_id not in colors:
				colors[cls_id] = (random.random(), random.random(), random.random())
			xmin = int(x0 * width)
			ymin = int(y0 * height)
			xmax = int(x1 * width)
			ymax = int(y1 * height)
			rect = plt.Rectangle((xmin, ymin), xmax - xmin,
								 ymax - ymin, fill=False,
								 edgecolor=colors[cls_id],
								 linewidth=3.5)
			plt.gca().add_patch(rect)
			class_name = str(cls_id)
			if classes and len(classes) > cls_id:
				class_name = classes[cls_id]
			plt.gca().text(xmin, ymin - 2,
							'{:s} {:.3f}'.format(class_name, score),
							bbox=dict(facecolor=colors[cls_id], alpha=0.5),
									fontsize=12, color='white')
			det_classes_and_score[class_name] = score

		plt.show()
		return det_classes_and_score


	def generate_random_image_file_from_manifest(self, bucket_name, test_manifest, manifest = 'validation'):
		
		root = manifest
		test_count = len(test_manifest)
		random_annotation = test_manifest[random.randint(0,test_count-1)]
		s3_uri = random_annotation['path']
		
		s3_key = os.path.basename(s3_uri)
		local_path = 'images/' + s3_key
	#     print(s3_key)
		s3.Bucket(bucket_name).download_file(
		root+s3_key, local_path)
		
		return local_path


	def download_image(self, url, to_save_filename='test-image.png'):

		print('downloading Image from {}'.format(url))
		response = requests.get(url, stream=True)
		with open(to_save_filename, 'wb') as out_file:
			shutil.copyfileobj(response.raw, out_file)
		print('Saved Image to {}'.format(to_save_filename)) 
		del response
		return to_save_filename






