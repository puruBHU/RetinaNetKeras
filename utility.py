import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def swap_xy(boxes):
	'''
	:param boxes: A tensor with shape '(num_boxes, 4)' representing bounding boxes
	:return: swapped boxes with same shape as that of input boxes
	'''
	return tf.stack([boxes[:,1], boxes[:,0], boxes[:,3], boxes[:,2]], axis=-1)

def convert_to_xywh(boxes):
	'''
	:param boxes: A tensor with shape '(num_boxes, 4)' representing bounding boxes
	:return:
	'''
	return tf.concat(
			(boxes[...,:2] + boxes[...,2:] / 2.0, boxes[...,2:] - boxes[...,:2]), axis = -1
			)

def convert_to_corner(boxes):
	'''
	:param boxes: A tensor with shape '(num_boxes, 4)' representing bounding boxes where boxes are of format
	'[x, y, width, height]'
	:return: A tensor of same shape as that of input tensor in format '[xtop, ytop, xbot, ybot]'
	'''
	return tf.concat([boxes[...,:2] - boxes[...,2:] / 2.0, boxes[...,:2] + boxes[...,2:] / 2.0], axis=-1)

def compute_iou(boxes1, boxes2):
	''':cvar
	'''
	boxes1_corners = convert_to_corner(boxes1)
	boxes2_corners = convert_to_corner(boxes2)

	lu = tf.maximum(boxes1_corners[:, None,:2], boxes2_corners[:,:2]) # left uppper (lu)
	rd = tf.minimum(boxes1_corners[:, None,2:], boxes2_corners[:,2:]) # right down

	intersection = tf.maximum(0.0, rd - lu)
	intersection_area = intersection[...,0] * intersection[...,1]
	boxes1_area = boxes1[:,2] * boxes1[:, 3]
	boxes2_area = boxes2[:,2] * boxes2[:, 3]

	union_area = tf.maximum(boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8)

	return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

def visualize_detection(image, boxes, classes, scores, figsize= (7,7),, linewidth=1, color=[0,0,1]):

	image = np.array(image, dtype = np.uint8)
	plt.figure(figsize = figsize)
	plt.axis('off')
	plt.imshow(image)
	ax = plt.gca()
	for box, _cls, score in zip(boxes, classes, scores):
		text = "{}: {:.2f}".format(_cls, score)
		x1, y1, x2, y2 = box
		w, h = x2 - x1, y2 - y1
		patch = plt.Rectangle([x1, y1], w,h, fill=False, edgecolor = color, linewidth=linewidth
		                      )
		ax.add_patch(patch)
		ax.text(x1,
		        y1,
		        text,
		        bbox={'facecolor':color, 'alpha':0.4},
		        clip_box=ax.clipbox,
		        clip_on=True
		        )
		plt.show()
		return ax



if __name__=='__main__':
	box1 = tf.convert_to_tensor([0,0,2,2], dtype = tf.float32)
	box2 = tf.convert_to_tensor([0,0, 3, 3], dtype = tf.float32)
	box1  = tf.reshape(box1,(1,4))
	box2  = tf.reshape(box2,(1,4))

	swap1      = swap_xy(box1)
	xywh_box1  = convert_to_xywh(box1)
	xy_corner1 = convert_to_corner(xywh_box1)
	print(swap1)
	print(xywh_box1)
	print(xy_corner1)


	#xywh_box1 = tf.expand_dims(xywh_box1, axis = 0)

	xywh_box2 = convert_to_xywh(box2)
	iou  = compute_iou(xywh_box1, xywh_box2)
	print(iou)