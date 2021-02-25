import tensorflow as tf
from tensorflow import keras

from utils import tf_utils


@tf.custom_gradient
def attr_mse_loss(label, pred_attr, dists, idx, thresh):

	def grad(dy, variables=None, *args):
		return None, grads, None, None, None

	mask = tf.cast(tf.where(dists <= thresh, tf.ones_like(pred_attr), tf.zeros_like(pred_attr)), tf.bool)

	label = tf.gather_nd(label, idx, batch_dims=1)

	with tf.GradientTape() as t:
		t.watch([pred_attr])
		loss = keras.losses.Huber()(label, pred_attr)

	grads = t.gradient(loss, pred_attr)
	grads = tf.where(mask, grads, tf.zeros_like(grads))

	return loss, grad


def xe_loss(label, pred, dist, idx, thresh=2, n_classes=9):

	clf = tf.gather_nd(label, idx, batch_dims=1)
	clf = tf.where(dist <= thresh, clf, tf.cast(tf.fill(clf.shape, n_classes-1), tf.int64))
	clf = tf.squeeze(clf, -1)

	weights = tf.where(clf == n_classes-1, 0.2, 0.8)

	return keras.losses.SparseCategoricalCrossentropy()(clf, pred, sample_weight=weights)


def chamfer_loss(label, pred):

	gt_p_dist, gt_p_idx, p_gt_dist, p_gt_idx = tf_utils.nn_distance(label, pred)

	chamfer_dist = tf.reduce_mean(gt_p_dist) + tf.reduce_mean(p_gt_dist)

	return chamfer_dist, tf.sqrt(p_gt_dist), p_gt_idx