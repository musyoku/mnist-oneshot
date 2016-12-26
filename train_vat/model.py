# -*- coding: utf-8 -*-
import math
import json, os, sys
from chainer import cuda
from args import args
sys.path.append(os.path.split(os.getcwd())[0])
from vat import VAT, Config
from sequential import Sequential
from sequential.layers import Linear, Merge, BatchNormalization, Gaussian
from sequential.functions import Activation, dropout, gaussian_noise, tanh, sigmoid

try:
	os.mkdir(args.model_dir)
except:
	pass

model_filename = args.model_dir + "/model.json"

if os.path.isfile(model_filename):
	print "loading", model_filename
	with open(model_filename, "r") as f:
		try:
			params = json.load(f)
		except Exception as e:
			raise Exception("could not load {}".format(model_filename))
else:
	config = Config()
	config.ndim_x = 28 * 28
	config.ndim_y = 10
	config.weight_init_std = 0.01
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0002
	config.momentum = 0.9
	config.gradient_clipping = 10
	config.weight_decay = 0
	config.lambda_ = 1
	config.Ip = 1

	model = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	model.add(Linear(None, 1200))
	model.add(Activation(config.nonlinearity))
	model.add(BatchNormalization(1200))
	model.add(Linear(None, 600))
	model.add(Activation(config.nonlinearity))
	model.add(BatchNormalization(600))
	model.add(Linear(None, config.ndim_y))

	params = {
		"config": config.to_dict(),
		"model": model.to_dict(),
	}

	with open(model_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

vat = VAT(params)
vat.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	vat.to_gpu()
