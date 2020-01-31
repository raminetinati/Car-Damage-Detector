import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sagemaker
from sagemaker.tensorflow import TensorFlowPredictor


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


	def display_test_img(self, bucket_name, test_manifest):
		
		root = 'training/'
		test_count = len(test_manifest)
		random_annotation = test_manifest[random.randint(0,test_count-1)]
		s3_uri = random_annotation['path']
		
		s3_key = os.path.basename(s3_uri)
		local_path = 'images/' + s3_key
	#     print(s3_key)
		s3.Bucket(bucket_name).download_file(
		root+s3_key, local_path)
		
		return s3_key, local_path


	def visualize_detection(self, bucket_name, test_manifest, thresh=0.6):

		s3_key, local_path = self.display_test_img(bucket_name, test_manifest)

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




