import numpy as np
import os, sys, time, math
from chainer import cuda
from chainer import functions as F
import pandas as pd
sys.path.append(os.path.split(os.getcwd())[0])
import dataset
from progress import Progress
from mnist_tools import load_train_images, load_test_images
from model import discriminator_params, generator_params, gan
from args import args

def main():
	# load MNIST images
	images, labels = dataset.load_train_images()

	# config
	discriminator_config = gan.config_discriminator
	generator_config = gan.config_generator

	# settings
	# _l -> labeled
	# _u -> unlabeled
	# _g -> generated
	max_epoch = 1000
	num_trains_per_epoch = 500
	plot_interval = 5
	batchsize_l = 100

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

	training_images_l, training_labels_l, training_images_u, validation_images, validation_labels = dataset.create_semisupervised(images, labels, num_validation_data, num_labeled_data, discriminator_config.ndim_output, seed=args.seed)
	print training_labels_l

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_supervised = 0

		for t in xrange(num_trains_per_epoch):
			# sample from data distribution
			images_l, label_onehot_l, label_ids_l = dataset.sample_labeled_data(training_images_l, training_labels_l, batchsize_l, discriminator_config.ndim_input, discriminator_config.ndim_output, binarize=False)

			# supervised loss
			py_x_l, activations_l = gan.discriminate(images_l, apply_softmax=False)
			loss_supervised = F.softmax_cross_entropy(py_x_l, gan.to_variable(label_ids_l))

			# update discriminator
			gan.backprop_discriminator(loss_supervised)

			sum_loss_supervised += float(loss_supervised.data)
			if t % 10 == 0:
				progress.show(t, num_trains_per_epoch, {})

		gan.save_discriminator(args.model_dir)

		# validation
		images_l, _, label_ids_l = dataset.sample_labeled_data(validation_images, validation_labels, num_validation_data, discriminator_config.ndim_input, discriminator_config.ndim_output, binarize=False)
		images_l_segments = np.split(images_l, num_validation_data // 500)
		label_ids_l_segments = np.split(label_ids_l, num_validation_data // 500)
		sum_accuracy = 0
		for images_l, label_ids_l in zip(images_l_segments, label_ids_l_segments):
			y_distribution, _ = gan.discriminate(images_l, apply_softmax=True, test=True)
			accuracy = F.accuracy(y_distribution, gan.to_variable(label_ids_l))
			sum_accuracy += float(accuracy.data)
		validation_accuracy = sum_accuracy / len(images_l_segments)
		
		progress.show(num_trains_per_epoch, num_trains_per_epoch, {
			"loss_l": sum_loss_supervised / num_trains_per_epoch,
			"accuracy": validation_accuracy,
		})

		# write accuracy to csv
		csv_results.append([epoch, validation_accuracy, progress.get_total_time()])
		data = pd.DataFrame(csv_results)
		data.columns = ["epoch", "accuracy", "min"]
		data.to_csv("{}/result.csv".format(args.model_dir))

if __name__ == "__main__":
	main()
