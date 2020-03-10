import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sagemaker
from sagemaker.tensorflow import TensorFlowPredictor
import requests
import shutil
import json
import numpy as np

class CarDamageDetector:


	def __init__(self, predictorEndpointName):
		print('Instantiating Car Damage Detector')
		if predictorEndpointName:
			self.predictor = sagemaker.RealTimePredictor(predictorEndpointName)
			self.set_class_labels()
		else:
			print('Must Supply an Endpoint Name')


	def set_class_labels(self):
		print('Adding Class Labels')

		self.object_categories = ['No Damage','Damage']

		print('Added {} Class Labels'.format(len(self.object_categories)))


	def test_model_on_manifest_data(self, bucket_name, manifest):

		image_path = self.generate_random_image_file_from_manifest(bucket_name, manifest)
		self.visualize_detection(image_path)

	def test_model_on_image(self, url_of_image):

		image_path = download_image(self, url, to_save_filename='test-image.png')
		self.visualize_detection(image_path)


	def predict_if_contains_damage(self, image_path):

		local_path = image_path
		with open(local_path, 'rb') as f:
			payload = f.read()
			payload = bytearray(payload)
		raw_img = mpimg.imread(local_path)
		plt.imshow(raw_img)
		self.predictor .content_type = 'application/x-image'
		result = json.loads(self.predictor.predict(payload))
		# the result will output the probabilities for all classes
		# find the class with maximum probability and print the class index
		index = np.argmax(result)
		object_categories = ['No Damage','Damage']
		print('Image Name {}'.format(local_path))
		print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))
		return result[index]


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







