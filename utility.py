import tensorflow as tf


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



if __name__=='__main__':
	box = tf.convert_to_tensor([0,0,2,2], dtype = tf.float32)
	box = tf.reshape(box,(1,4))

	swap      = swap_xy(box)
	xywh_box = convert_to_xywh(box)
	xy_corner = convert_to_corner(xywh_box)
	print(swap)
	print(xywh_box)
	print(xy_corner)