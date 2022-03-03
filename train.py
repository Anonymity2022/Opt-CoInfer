import importlib
import os
import time
import random
import math

import torch
from torch import multiprocessing
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from utils.model_profiling import model_profiling
from utils.transforms import Lighting
from utils.distributed import init_dist, master_only, is_master
from utils.distributed import get_rank, get_world_size
from utils.distributed import dist_all_reduce_tensor
from utils.distributed import master_only_print as print
from utils.distributed import AllReduceDistributedDataParallel, allreduce_grads
from utils.loss_ops import CrossEntropyLossSoft, CrossEntropyLossSmooth
from utils.config import FLAGS
from utils.meters import ScalarMeter, flush_scalar_meters
import argparse
import warnings

def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    z_architecture_list=[64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M']
    vgg_layer_id=[2,5,6,9,12,13,16,19,22,23,26,29,32,33,36,39,42,43]
    vgg_id=vgg_layer_id[FLAGS.z_partition_id]
    z_architecture_list[FLAGS.z_partition_id]=FLAGS.z_pruning_num
    model = model_lib.vgg16_bn(z_architecture_list,vgg_id, FLAGS.z_quantization)
    if getattr(FLAGS, 'distributed', False):
        gpu_id = init_dist()
        if getattr(FLAGS, 'distributed_all_reduce', False):
            # seems faster
            model_wrapper = AllReduceDistributedDataParallel(model.cuda())
        else:
            model_wrapper = torch.nn.parallel.DistributedDataParallel(
                model.cuda(), [gpu_id], gpu_id)
    else:
        model_wrapper = torch.nn.DataParallel(model).cuda()
    return model, model_wrapper


def data_transforms():
    """get transform of dataset"""
    if FLAGS.data_transforms in [
            'imagenet1k_basic', 'imagenet1k_inception', 'imagenet1k_mobile']:
        if FLAGS.data_transforms == 'imagenet1k_inception':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_basic':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_mobile':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.25
            jitter_param = 0.4
            lighting_param = 0.1
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            Lighting(lighting_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    else:
        try:
            transforms_lib = importlib.import_module(FLAGS.data_transforms)
            return transforms_lib.data_transforms()
        except ImportError:
            raise NotImplementedError(
                'Data transform {} is not yet implemented.'.format(
                    FLAGS.data_transforms))
    return train_transforms, val_transforms, test_transforms


def dataset(train_transforms, val_transforms, test_transforms):
    """get dataset for classification"""
    if FLAGS.dataset == 'imagenet1k':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    else:
        try:
            dataset_lib = importlib.import_module(FLAGS.dataset)
            return dataset_lib.dataset(
                train_transforms, val_transforms, test_transforms)
        except ImportError:
            raise NotImplementedError(
                'Dataset {} is not yet implemented.'.format(FLAGS.dataset_dir))
    return train_set, val_set, test_set


def data_loader(train_set, val_set, test_set):
    """get data loader"""
    train_loader = None
    val_loader = None
    test_loader = None
    # infer batch size
    if getattr(FLAGS, 'batch_size', False):
        if getattr(FLAGS, 'batch_size_per_gpu', False):
            assert FLAGS.batch_size == (
                FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job)
        else:
            assert FLAGS.batch_size % FLAGS.num_gpus_per_job == 0
            FLAGS.batch_size_per_gpu = (
                FLAGS.batch_size // FLAGS.num_gpus_per_job)
    elif getattr(FLAGS, 'batch_size_per_gpu', False):
        FLAGS.batch_size = FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job
    else:
        raise ValueError('batch size (per gpu) is not defined')
    batch_size = int(FLAGS.batch_size/get_world_size())
    if FLAGS.data_loader == 'imagenet1k_basic':
        if getattr(FLAGS, 'distributed', False):
            if FLAGS.test_only:
                train_sampler = None
            else:
                train_sampler = DistributedSampler(train_set)
            val_sampler = DistributedSampler(val_set)
        else:
            train_sampler = None
            val_sampler = None
        if not FLAGS.test_only:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                pin_memory=True,
                num_workers=FLAGS.data_loader_workers,
                drop_last=getattr(FLAGS, 'drop_last', False))
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=FLAGS.data_loader_workers,
            drop_last=getattr(FLAGS, 'drop_last', False))
        test_loader = val_loader
    else:
        try:
            data_loader_lib = importlib.import_module(FLAGS.data_loader)
            return data_loader_lib.data_loader(train_set, val_set, test_set)
        except ImportError:
            raise NotImplementedError(
                'Data loader {} is not yet implemented.'.format(
                    FLAGS.data_loader))
    if train_loader is not None:
        FLAGS.data_size_train = len(train_loader.dataset)
    if val_loader is not None:
        FLAGS.data_size_val = len(val_loader.dataset)
    if test_loader is not None:
        FLAGS.data_size_test = len(test_loader.dataset)
    return train_loader, val_loader, test_loader


def get_lr_scheduler(optimizer):
    """get learning rate"""
    warmup_epochs = getattr(FLAGS, 'lr_warmup_epochs', 0)
    if FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma)
    elif FLAGS.lr_scheduler == 'exp_decaying':
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            if i == 0:
                lr_dict[i] = 1
            else:
                lr_dict[i] = lr_dict[i-1] * FLAGS.exp_decaying_lr_gamma
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'linear_decaying':
        num_epochs = FLAGS.num_epochs - warmup_epochs
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = 1. - (i - warmup_epochs) / num_epochs
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'cosine_decaying':
        num_epochs = FLAGS.num_epochs - warmup_epochs
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = (
                1. + math.cos(
                    math.pi * (i - warmup_epochs) / num_epochs)) / 2.
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    else:
        try:
            lr_scheduler_lib = importlib.import_module(FLAGS.lr_scheduler)
            return lr_scheduler_lib.get_lr_scheduler(optimizer)
        except ImportError:
            raise NotImplementedError(
                'Learning rate scheduler {} is not yet implemented.'.format(
                    FLAGS.lr_scheduler))
    return lr_scheduler


def get_optimizer(model):
    """get optimizer"""
    if FLAGS.optimizer == 'sgd':
        # all depthwise convolution (N, 1, x, x) has no weight decay
        # weight decay only on normal conv and fc
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        try:
            optimizer_lib = importlib.import_module(FLAGS.optimizer)
            return optimizer_lib.get_optimizer(model)
        except ImportError:
            raise NotImplementedError(
                'Optimizer {} is not yet implemented.'.format(FLAGS.optimizer))
    return optimizer


def set_random_seed(seed=None):
    """set random seed"""
    if seed is None:
        seed = getattr(FLAGS, 'random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@master_only
def get_meters(phase):
    """util function for meters"""
    def get_single_meter(phase, suffix=''):
        meters = {}
        meters['loss'] = ScalarMeter('{}_loss/{}'.format(phase, suffix))
        for k in FLAGS.topk:
            meters['top{}_error'.format(k)] = ScalarMeter(
                '{}_top{}_error/{}'.format(phase, k, suffix))
        if phase == 'train':
            meters['lr'] = ScalarMeter('learning_rate')
        return meters

    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    meters = {}
    for width_mult in FLAGS.width_mult_list:
            meters[str(width_mult)] = get_single_meter(phase, str(width_mult))
    if phase == 'val':
        meters['best_val'] = ScalarMeter('best_val')
    return meters


@master_only
def profiling(model, use_cuda):
	"""profiling on either gpu or cpu"""
	print('Start model profiling, use_cuda: {}.'.format(use_cuda))
	for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
		model.apply(
			lambda m: setattr(m, 'width_mult', width_mult))
		print('Model profiling with width mult {}x:'.format(width_mult))
		flops, params = model_profiling(
			model, FLAGS.image_size, FLAGS.image_size, use_cuda=use_cuda,
			verbose=getattr(FLAGS, 'profiling_verbose', False))
	return flops, params


def lr_schedule_per_iteration(optimizer, epoch, batch_idx=0):
    """ function for learning rate scheuling per iteration """
    warmup_epochs = getattr(FLAGS, 'lr_warmup_epochs', 0)
    num_epochs = FLAGS.num_epochs - warmup_epochs
    iters_per_epoch = FLAGS.data_size_train / FLAGS.batch_size
    current_iter = epoch * iters_per_epoch + batch_idx + 1
    if getattr(FLAGS, 'lr_warmup', False) and epoch < warmup_epochs:
        linear_decaying_per_step = FLAGS.lr/warmup_epochs/iters_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_iter * linear_decaying_per_step
    elif FLAGS.lr_scheduler == 'linear_decaying':
        linear_decaying_per_step = FLAGS.lr/num_epochs/iters_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] -= linear_decaying_per_step
    elif FLAGS.lr_scheduler == 'cosine_decaying':
        mult = (
            1. + math.cos(
                math.pi * (current_iter - warmup_epochs * iters_per_epoch)
                / num_epochs / iters_per_epoch)) / 2.
        for param_group in optimizer.param_groups:
            param_group['lr'] = FLAGS.lr * mult
    else:
        pass


def forward_loss(
        model, criterion, input, target, meter, soft_target=None,
        soft_criterion=None, return_soft_target=False, return_acc=False):
    """forward model and return loss"""
    output = model(input)
    if soft_target is not None:
        loss = torch.mean(soft_criterion(output, soft_target))
    else:
        loss = torch.mean(criterion(output, target))
    # topk
    _, pred = output.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = []
    for k in FLAGS.topk:
        correct_k.append(correct[:k].float().sum(0))
    tensor = torch.cat([loss.view(1)] + correct_k, dim=0)
    # allreduce
    tensor = dist_all_reduce_tensor(tensor)
    # cache to meter
    tensor = tensor.cpu().detach().numpy()
    bs = (tensor.size-1)//2
    for i, k in enumerate(FLAGS.topk):
        error_list = list(1.-tensor[1+i*bs:1+(i+1)*bs])
        if return_acc and k == 1:
            top1_error = sum(error_list) / len(error_list)
            return loss, top1_error
        if meter is not None:
            meter['top{}_error'.format(k)].cache_list(error_list)
    if meter is not None:
        meter['loss'].cache(tensor[0])
    if return_soft_target:
        return loss, torch.nn.functional.softmax(output, dim=1)
    return loss


def run_one_epoch(
		epoch, loader, model, criterion, optimizer, meters, phase='train',
		soft_criterion=None):
	"""run one epoch for train/val/test/cal"""
	t_start = time.time()
	assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
	train = phase == 'train'
	if train:
		model.train()
	else:
		model.eval()
		if phase == 'cal':
			model.apply(bn_calibration_init)


	if getattr(FLAGS, 'distributed', False):
		loader.sampler.set_epoch(epoch)
	for batch_idx, (input, target) in enumerate(loader):
		if phase == 'cal':
			if batch_idx == getattr(FLAGS, 'bn_cal_batch_num', -1):
				break
		target = target.cuda(non_blocking=True)
		if train:
			# change learning rate if necessary
			lr_schedule_per_iteration(optimizer, epoch, batch_idx)
			optimizer.zero_grad()
			widths_train = FLAGS.width_mult_list
			for width_mult in widths_train:
				model.apply(
						lambda m: setattr(m, 'width_mult', width_mult))
				meter = meters[str(width_mult)]
				loss = forward_loss(
							model, criterion, input, target, meter)
				loss.backward()
			if (getattr(FLAGS, 'distributed', False)
					and getattr(FLAGS, 'distributed_all_reduce', False)):
				allreduce_grads(model)
			optimizer.step()
			if is_master():
				for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
					meter = meters[str(width_mult)]
					meter['lr'].cache(optimizer.param_groups[0]['lr'])
			else:
				pass
		else:
			for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
				model.apply(
					lambda m: setattr(m, 'width_mult', width_mult))
				if is_master():
					meter = meters[str(width_mult)]
				else:
					meter = None
				forward_loss(model, criterion, input, target, meter)
	if is_master():
		for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
			results = flush_scalar_meters(meters[str(width_mult)])
			# print('{:.1f}s\t{}\t{}\t{}/{}: '.format(
				# time.time() - t_start, phase, str(width_mult), epoch,
				# FLAGS.num_epochs) + ', '.join(
					 # '{}: {:.3f}'.format(k, v) for k, v in results.items()))
	# elif is_master():
		# results = flush_scalar_meters(meters)
		# print(
			# '{:.1f}s\t{}\t{}/{}: '.format(
				# time.time() - t_start, phase, epoch, FLAGS.num_epochs) +
			# ', '.join('{}: {:.3f}'.format(k, v) for k, v in results.items()))
	else:
		results = None
	return results


def train_val_test():
    """train and val"""
    torch.backends.cudnn.benchmark = True
    # seed
    set_random_seed()

    # model
    model, model_wrapper = get_model()
    if getattr(FLAGS, 'label_smoothing', 0):
        criterion = CrossEntropyLossSmooth(reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    if getattr(FLAGS, 'inplace_distill', True):
        soft_criterion = CrossEntropyLossSoft(reduction='none')
    else:
        soft_criterion = None
    # check pretrained
    if getattr(FLAGS, 'pretrained', False):
        checkpoint = torch.load(
            FLAGS.pretrained, map_location=lambda storage, loc: storage)
        # update keys from external models
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        model_wrapper.load_state_dict(checkpoint)
        print('Loaded model {}.'.format(FLAGS.pretrained))

    optimizer = get_optimizer(model_wrapper)

    # check resume training
    if os.path.exists(os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt')):
        checkpoint = torch.load(
            os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'),
            map_location=lambda storage, loc: storage)
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = get_lr_scheduler(optimizer)
        lr_scheduler.last_epoch = last_epoch
        best_val = checkpoint['best_val']
        train_meters, val_meters = checkpoint['meters']
#        print('Loaded checkpoint {} at epoch {}.'.format(
#            FLAGS.log_dir, last_epoch))
    else:
        lr_scheduler = get_lr_scheduler(optimizer)
        last_epoch = lr_scheduler.last_epoch
        best_val = 1.
        train_meters = get_meters('train')
        val_meters = get_meters('val')
        # if start from scratch, print model and do profiling
        #print(model_wrapper)

    # data
    train_transforms, val_transforms, test_transforms = data_transforms()
    train_set, val_set, test_set = dataset(
        train_transforms, val_transforms, test_transforms)
    train_loader, val_loader, test_loader = data_loader(
        train_set, val_set, test_set)


    if getattr(FLAGS, 'test_only', False) and (test_loader is not None):
        print('Start testing.')
#        model_wrapper.load_state_dict(torch.load('logs/1.pth'))
        test_meters = get_meters('test')
        with torch.no_grad():
            for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                model_wrapper.apply(
                        lambda m: setattr(m, 'width_mult', width_mult))
                run_one_epoch(
                        last_epoch, test_loader, model_wrapper, criterion,
                        optimizer, test_meters, phase='test')
        return

    if getattr(FLAGS, 'nonuniform_diff_seed', False):
        set_random_seed(getattr(FLAGS, 'random_seed', 0) + get_rank())

#    print('Start training.')
    for epoch in range(last_epoch+1, FLAGS.num_epochs):
        if getattr(FLAGS, 'skip_training', False):
            print('Skip training at epoch: {}'.format(epoch))
            break
        lr_scheduler.step()
        # train
        results = run_one_epoch(
            epoch, train_loader, model_wrapper, criterion, optimizer,
            train_meters, phase='train', soft_criterion=soft_criterion)

        # val
        if val_meters is not None:
            val_meters['best_val'].cache(best_val)
        if epoch>0.6*FLAGS.num_epochs:
            with torch.no_grad():
                results = run_one_epoch(
                epoch, val_loader, model_wrapper, criterion, optimizer,
                val_meters, phase='val')
            if is_master() and results['top1_error'] < best_val:
                best_val = results['top1_error']
                torch.save(
                {
                    'model': model_wrapper.state_dict(),
                },
                os.path.join(FLAGS.log_dir, 'best_model.pt'))
                print('New best validation top1 error: {:.3f}'.format(best_val))
        # save latest checkpoint
        # if is_master():
            # torch.save(
                # {
                    # 'model': model_wrapper.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'last_epoch': epoch,
                    # 'best_val': best_val,
                    # 'meters': (train_meters, val_meters),
                # },
                # os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'))
    with open(FLAGS.data_base_address,'a') as f:
        id=[0,1,3,4,6,7,8,10,11,12,14,15,16]
        f.write('{:.0f}+{:.0f}-{:.0f}*{:.3f} \n'.format(id.index(FLAGS.z_partition_id),FLAGS.z_pruning_num,FLAGS.z_quantization,best_val))
    return


def init_multiprocessing():
    # print(multiprocessing.get_start_method())
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass


def main():
    """train and eval model"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    warnings.filterwarnings('ignore')
    init_multiprocessing()
    train_val_test()


if __name__ == "__main__":
    main()
