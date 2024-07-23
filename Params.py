import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--att_layer', default=4, type=int, help='layer number of multi att')
	parser.add_argument('--att_size', default=12000, type=int, help='max size of multi att')
	parser.add_argument('--batch', default=512, type=int, help='batch size')
	parser.add_argument('--data', default='gowalla', type=str, help='name of dataset')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')
	parser.add_argument('--divSize', default=10000, type=int, help='div size for smallTestEpoch')
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--graphNum', default=8, type=int, help='number of graphs based on time series')
	parser.add_argument('--graphSampleN', default=15000, type=int, help='use 25000 for training and 200000 for testing, empirically')
	parser.add_argument('--hyperNum', default=128, type=int, help='number of hyper edges')
	parser.add_argument('--hyperReg', default=1e-4, type=float, help='regularizer for hyper connections')
	parser.add_argument('--keepRate', default=0.5, type=float, help='rate for dropout')
	parser.add_argument('--latdim', default=64, type=int, help='embedding size')
	parser.add_argument('--leaky', default=0.5, type=float, help='slope for leaky relu')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--memosize', default=2, type=int, help='memory size')
	parser.add_argument('--mult', default=100, type=float, help='multiplier for the result')
	parser.add_argument('--nfs', default=False, type=bool, help='load from nfs')
	parser.add_argument('--percent', default=0.0, type=float, help='percent of noise for noise robust test')
	parser.add_argument('--pos_length', default=200, type=int, help='max length of a sequence')
	parser.add_argument('--pred_num', default=5, type=int, help='pred number of train')
	parser.add_argument('--query_vector_dim', type=int, default=64, help='number of query vector\'s dimension [default: 64]')
	parser.add_argument('--rank', default=4, type=int, help='embedding size')
	parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
	parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	parser.add_argument('--slot', default=1, type=float, help='length of time slots')
	parser.add_argument('--ssl', default=True, type=bool, help='use self-supervised learning')
	parser.add_argument('--sslNum', default=20, type=int, help='batch size for ssl')
	parser.add_argument('--ssl_reg', default=1e-4, type=float, help='reg weight for ssl loss')
	parser.add_argument('--ssldim', default=32, type=int, help='user weight embedding size')
	parser.add_argument('--subUsrDcy', default=0.9, type=float, help='decay factor for sub-users over time')
	parser.add_argument('--subUsrSize', default=10, type=int, help='number of item for each sub-user')
	parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
	parser.add_argument('--temp', default=1, type=float, help='temperature in ssl loss')
	parser.add_argument('--test', default=True, type=bool, help='test or val')
	parser.add_argument('--testSize', default=100, type=int, help='size for test')
	parser.add_argument('--testbatch', default=64, type=int, help='test batch size')
	parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
	parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
	parser.add_argument('--uid', default=0, type=int, help='show user score')
	parser.add_argument('--num_attention_heads', type=int, default=16, help='number of num attention heads [default: 16]')
	parser.add_argument('--log_dir', type=str, default='output', help='Directory to save results')

	return parser.parse_args()
args = parse_args()
args.decay_step = args.trnNum//args.batch