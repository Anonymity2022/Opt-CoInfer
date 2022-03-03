import torch as t
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-z_num', type=int, default=0)
parser.add_argument('-gpu', type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if os.path.isfile('./logs/best_model.pt'):
	checkpoint = t.load('./logs/best_model.pt', map_location=lambda storage, loc: storage)
	checkpoint=checkpoint['model']
	checkpoint2={}
	z_compression=args.z_num
	for k,v in checkpoint.items():
		if k=='module.features.7.weight' or k=='module.features.7.bias' or k=='module.features.8.weight' or k=='module.features.8.bias' or k=='module.features.8.running_var' or k=='module.features.8.running_mean':
			checkpoint2[k]=v[0:z_compression]
		elif k=='module.features.10.weight':
			checkpoint2[k]=v[:,0:z_compression]
		else:
			checkpoint2[k]=v
	t.save(checkpoint2,'./logs/best_model.pt')
