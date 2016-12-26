import numpy as np
import os, sys, time
from chainer import cuda
from chainer import functions as F
import pandas as pd
from model import aae
from progress import Progress
from args import args
import dataset
import sampler

def main():
	# load MNIST images
	images, labels = dataset.load_train_images()

	# config
	config = aae.config

	# settings
	# _l -> labeled
	# _u -> unlabeled
	max_epoch = 1000
	num_trains_per_epoch = 500
	batchsize_l = 100
	batchsize_u = 100
	alpha = 1

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# save validation accuracy per epoch
	csv_results = []

	# create semi-supervised split
	num_validation_data = 10000
	num_labeled_data = args.num_labeled
	if batchsize_l > num_labeled_data:
		batchsize_l = num_labeled_data
	num_types_of_label = 10
	training_images_l, training_labels_l, training_images_u, validation_images, validation_labels = dataset.create_semisupervised(images, labels, num_validation_data, num_labeled_data, num_types_of_label, seed=args.seed)
	print training_labels_l

	# classification
	# 0 -> true sample
	# 1 -> generated sample
	class_true = aae.to_variable(np.zeros(batchsize_u, dtype=np.int32))
	class_fake = aae.to_variable(np.ones(batchsize_u, dtype=np.int32))

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_supervised = 0

		for t in xrange(num_trains_per_epoch):
			# sample from data distribution
			images_l, label_onehot_l, label_ids_l = dataset.sample_labeled_data(training_images_l, training_labels_l, batchsize_l, config.ndim_x, config.ndim_y, normalize_zero_to_one=True)

			# supervised phase
			unnormalized_q_y_x_l, z_l = aae.encode_x_yz(images_l, apply_softmax=False)
			loss_supervised = F.softmax_cross_entropy(unnormalized_q_y_x_l, aae.to_variable(label_ids_l))
			aae.backprop_generator(loss_supervised)

			sum_loss_supervised += float(loss_supervised.data)

			if t % 10 == 0:
				progress.show(t, num_trains_per_epoch, {})

		aae.save_generator(args.model_dir)

		# validation phase
		# split validation data to reduce gpu memory consumption
		images_v, _, label_ids_v = dataset.sample_labeled_data(validation_images, validation_labels, num_validation_data, config.ndim_x, config.ndim_y, normalize_zero_to_one=True)
		images_v_segments = np.split(images_v, num_validation_data // 500)
		label_ids_v_segments = np.split(label_ids_v, num_validation_data // 500)
		num_correct = 0
		for images_v, labels_v in zip(images_v_segments, label_ids_v_segments):
			predicted_labels = aae.argmax_x_label(images_v, test=True)
			for i, label in enumerate(predicted_labels):
				if label == labels_v[i]:
					num_correct += 1
		validation_accuracy = num_correct / float(num_validation_data)
		
		progress.show(num_trains_per_epoch, num_trains_per_epoch, {
			"loss_s": sum_loss_supervised / num_trains_per_epoch,
			"accuracy": validation_accuracy
		})

		# write accuracy to csv
		csv_results.append([epoch, validation_accuracy])
		data = pd.DataFrame(csv_results)
		data.columns = ["epoch", "accuracy"]
		data.to_csv("{}/result.csv".format(args.model_dir))

if __name__ == "__main__":
	main()
