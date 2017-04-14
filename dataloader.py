# dataloader.py

import math

import torch
import datasets
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

import utils as utils

class Dataloader:

	def __init__(self, args):
		self.args = args

		self.loader_input = args.loader_input
		self.loader_label = args.loader_label

		self.split_test = args.split_test
		self.split_train = args.split_train
		self.dataset_test = args.dataset_test
		self.dataset_train = args.dataset_train
		self.resolution = (args.resolution_wide, args.resolution_high)

		self.input_filename_test = args.input_filename_test
		self.label_filename_test = args.label_filename_test
		self.input_filename_train = args.input_filename_train
		self.label_filename_train = args.label_filename_train

		if self.dataset_train == 'lsun':
			self.dataset_train = datasets.LSUN(db_path=args.dataroot, classes=['bedroom_train'],
				transform=transforms.Compose([
					transforms.Scale(self.resolution),
					transforms.CenterCrop(self.resolution),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					])
				)
		
		elif self.dataset_train == 'cifar10':
			self.dataset_train = datasets.CIFAR10(root=self.args.dataroot, train=True, download=True,
				transform=transforms.Compose([
					transforms.Scale(self.resolution),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					])
				)
		
		elif self.dataset_train == 'filelist':
			self.dataset_train = datasets.FileList(self.input_filename_train, self.label_filename_train, self.split_train,
				self.split_test, train=True,
				transform_train=transforms.Compose([
					transforms.Scale(self.resolution),
					transforms.CenterCrop(self.resolution),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					]),
				transform_test=transforms.Compose([
					transforms.Scale(self.resolution),
					transforms.CenterCrop(self.resolution),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					]),
				loader_input=self.loader_input,
				loader_label=self.loader_label,
				)

		elif self.dataset_train == 'folderlist':
			self.dataset_train = datasets.FileList(self.input_filename_train, self.label_filename_train, self.split_train,
				self.split_test, train=True,
				transform_train=transforms.Compose([
					transforms.Scale(self.resolution),
					transforms.CenterCrop(self.resolution),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					]),
				transform_test=transforms.Compose([
					transforms.Scale(self.resolution),
					transforms.CenterCrop(self.resolution),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					]),
				loader_input=self.loader_input,
				loader_label=self.loader_label,
				)

		else:
			raise(Exception("Unknown Dataset"))

		if self.dataset_test == 'lsun':
			self.dataset_val = datasets.LSUN(db_path=args.dataroot, classes=['bedroom_val'],
				transform=transforms.Compose([
					transforms.Scale(self.resolution),
					transforms.CenterCrop(self.resolution),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					])
				)
		
		elif self.dataset_test == 'cifar10':
			self.dataset_val = datasets.CIFAR10(root=self.args.dataroot, train=False, download=True,
				transform=transforms.Compose([
					transforms.Scale(self.resolution),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					])
				)

		elif self.dataset_test == 'filelist':
			self.dataset_test = datasets.FileList(self.input_filename_test, self.label_filename_test, self.split_train,
				self.split_test, train=True,
				transform_train=transforms.Compose([
					transforms.Scale(self.resolution),
					transforms.CenterCrop(self.resolution),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					]),
				loader_input=self.loader_input,
				loader_label=self.loader_label,
				)

		elif self.dataset_test == 'folderlist':
			self.dataset_test = datasets.FileList(self.input_filename_test, self.label_filename_test, self.split_train,
				self.split_test, train=True,
				transform_train=transforms.Compose([
					transforms.Scale(self.resolution),
					transforms.CenterCrop(self.resolution),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					]),
				loader_input=self.loader_input,
				loader_label=self.loader_label,
				)
			
		else:
			raise(Exception("Unknown Dataset"))

	def create(self, flag=None):
		if flag == "Train":
			dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.args.batch_size,
				shuffle=True, num_workers=int(self.args.nthreads))
			return dataloader_train

		if flag == "Test":
			dataloader_test = torch.utils.data.DataLoader(self.dataset_val, batch_size=self.args.batch_size,
				shuffle=True, num_workers=int(self.args.nthreads))
			return dataloader_test

		if flag == None:
			dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.args.batch_size,
				shuffle=True, num_workers=int(self.args.nthreads))
		
			dataloader_test = torch.utils.data.DataLoader(self.dataset_val, batch_size=self.args.batch_size,
				shuffle=True, num_workers=int(self.args.nthreads))
			return dataloader_train, dataloader_test