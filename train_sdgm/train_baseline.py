import numpy as np
import os, sys, time
from chainer import cuda
from chainer import functions as F
import pandas as pd
sys.path.append(os.path.split(os.getcwd())[0])
import dataset
from progress import Progress
from model import sdgm
from args import args

def main():
	# load MNIST images
	images, labels = dataset.load_train_images()

	# config
	config = sdgm.config

	# settings
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

	# init weightnorm layers
	if config.use_weightnorm:
		print "initializing weight normalization layers ..."
		images_l, label_onehot_l, label_id_l = dataset.sample_labeled_data(training_images_l, training_labels_l, batchsize_l, config.ndim_x, config.ndim_y, normalize_zero_to_one=True)
		images_u = dataset.sample_unlabeled_data(training_images_u, batchsize_u, config.ndim_x, normalize_zero_to_one=True)
		sdgm.compute_lower_bound(images_l, label_onehot_l, images_u)

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch):
		progress.start_epoch(epoch, max_epoch)
		sum_lower_bound_l = 0
		sum_loss_classifier = 0

		for t in xrange(num_trains_per_epoch):
			# sample from data distribution
			images_l, label_onehot_l, label_ids_l = dataset.sample_labeled_data(training_images_l, training_labels_l, batchsize_l, config.ndim_x, config.ndim_y, normalize_zero_to_one=True)
			images_u = dataset.sample_unlabeled_data(training_images_u, 0, config.ndim_x, normalize_zero_to_one=True)

			# lower bound loss 
			lower_bound, lb_labeled, lb_unlabeled = sdgm.compute_lower_bound(images_l, label_onehot_l, images_u)
			loss_lower_bound = -lower_bound

			# classification loss
			a_l = sdgm.encode_x_a(images_l, False)
			unnormalized_y_distribution = sdgm.encode_ax_y_distribution(a_l, images_l, softmax=False)
			loss_classifier = alpha * F.softmax_cross_entropy(unnormalized_y_distribution, sdgm.to_variable(label_ids_l))

			# backprop
			sdgm.backprop(loss_classifier + loss_lower_bound)

			sum_lower_bound_l += float(lb_labeled.data)
			sum_loss_classifier += float(loss_classifier.data)
			progress.show(t, num_trains_per_epoch, {})

		sdgm.save(args.model_dir)

		# validation
		images_l, _, label_ids_l = dataset.sample_labeled_data(validation_images, validation_labels, num_validation_data, config.ndim_x, config.ndim_y, normalize_zero_to_one=True)
		images_l_segments = np.split(images_l, num_validation_data // 500)
		label_ids_l_segments = np.split(label_ids_l, num_validation_data // 500)
		sum_accuracy = 0
		for images_l, label_ids_l in zip(images_l_segments, label_ids_l_segments):
			y_distribution = sdgm.encode_x_y_distribution(images_l, softmax=True, test=True)
			accuracy = F.accuracy(y_distribution, sdgm.to_variable(label_ids_l))
			sum_accuracy += float(accuracy.data)
		validation_accuracy = sum_accuracy / len(images_l_segments)
		
		progress.show(num_trains_per_epoch, num_trains_per_epoch, {
			"lb_l": sum_lower_bound_l / num_trains_per_epoch,
			"loss_spv": sum_loss_classifier / num_trains_per_epoch,
			"accuracy": validation_accuracy,
		})

		# write accuracy to csv
		csv_results.append([epoch, validation_accuracy, progress.get_total_time()])
		data = pd.DataFrame(csv_results)
		data.columns = ["epoch", "accuracy", "min"]
		data.to_csv("{}/result.csv".format(args.model_dir))

if __name__ == "__main__":
	main()
