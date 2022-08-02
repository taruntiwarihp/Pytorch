from .model import *
from .efficientnet import build_efficientnet_model
from .att_resnet import AttentionResnet

def create_model(opts):
	if opts.model_type == 'base':
		model = HairClassificationModelBase(opts.n_class)
		opts.img_dim = 224

	elif opts.model_type == 'wide_res':
		model = HairClassificationModelWSR(opts.n_class)
		opts.img_dim = 224

	elif opts.model_type == 'mobile_net':
		model = HairClassificationModelMobile(opts.n_class)
		opts.img_dim = 224
		
	elif opts.model_type == 'xception':
		model = HairClassificationModelXception(opts.n_class)
		opts.img_dim = 299

	elif opts.model_type == 'vgg':
		model = HairClassificationModelVGG(opts.n_class)
		opts.img_dim = 224

	elif opts.model_type == 'densenet':
		model = HairClassificationModelDensenet(opts.n_class)
		opts.img_dim = 224

	elif opts.model_type == 'squeezenet':
		model = HairClassificationModelSqueezenet(opts.n_class)
		opts.img_dim = 224
        
	elif 'efficientnet' in opts.model_type:
		model_name = opts.model_type.replace('_', '-')
		model = build_efficientnet_model(model_name, num_classes=opts.n_class)
		opts.img_dim = 224

	elif opts.model_type == 'attentionresnet':
		model = AttentionResnet(opts.n_class)
		opts.img_dim = 224

	else:
		raise NotImplementedError('Not found')

	return model
