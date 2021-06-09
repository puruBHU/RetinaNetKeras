import numpy as np
import tensorflow as tf
from utility import swap_xy, convert_to_xywh

def random_flip_horizontal(image, boxes):
	if tf.random.uniform([], 0, 1) > 0.5:
		image = tf.image.flip_left_right(image)
		boxes = tf.stack(
				[1 - boxes[:,2], boxes[:,1], 1 - boxes[:,0], boxes[:3]], axis=-1
				)
		return image, boxes

def resize_and_pad_image(image, min_side=800, max_side=1333, jitter=[640, 1024], stride=128):
	image_shape = tf.cast(tf.shape(image)[:2], dtype =tf.float32)

	if jitter is not None:
		min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
	ratio = min_side / tf.reduce_min(image_shape)
	if ratio * tf.reduce_max(image_shape) > max_side:
		ratio = max_side / tf.reduce_max(image_shape)

	image_shape = ratio * image_shape

	image = tf.image.resize(image, tf.cast(image_shape, dtype = tf.int32))
	padded_image_shape = tf.cast(
			tf.math.ceil(image_shape / stride) * stride, dtype= tf.int32
			)

	image = tf.image.pad_to_bounding_box(image, 0, 0, padded_image_shape[0],
	                                     padded_image_shape[1]
	                                     )

	return image, image_shape, ratio

def preprocess_data(sample):

	image = sample["image"]
	bbox = swap_xy(sample["objects"]["bbox"])
	class_id = tf.cast(sample["objects"]["label"], dtype = tf.int32)

	image, bbox = random_flip_horizontal(image, bbox)
	image, image_shape, _ = resize_and_pad_image(image)

	bbox = tf.stack(
			[
				bbox[:,0] * image_shape[1],
				bbox[:,1] * image_shape[0],
				bbox[:,2] * image_shape[1],
				bbox[:,3] * image_shape[0]
				]
			axis = -1
	)
	bbox = convert_to_xywh(bbox)
	return image, bbox, class_id
