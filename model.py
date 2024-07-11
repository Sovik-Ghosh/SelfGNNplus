from ast import arg
from curses import meta
from site import USER_BASE
from matplotlib.cbook import silent_list
from Params import args
import utils.NNLayers as NNs
from utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import utils.TimeLogger as logger
import numpy as np
from utils.TimeLogger import log
from DataHandler import negSamp,negSamp_fre, transpose, DataHandler, transToLsts
from utils.attention import AdditiveAttention,MultiHeadSelfAttention
import scipy.sparse as sp
from random import randint

class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train_' + met] = list()
			self.metrics['Test_' + met] = list()
	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret
	def prepareModel(self):
		self.keepRate = tf.placeholder(dtype=tf.float32, shape=[])
		self.is_train = tf.placeholder_with_default(True, (), 'is_train')
		NNs.leaky = args.leaky
		self.actFunc = 'leakyRelu'
		adj = self.handler.trnMat
		idx, data, shape = transToLsts(adj, norm=True)
		self.adj = tf.sparse.SparseTensor(idx, data, shape)
		self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])
		self.sequence = tf.placeholder(name='sequence', dtype=tf.int32, shape=[args.batch,args.pos_length])
		self.mask = tf.placeholder(name='mask', dtype=tf.float32, shape=[args.batch,args.pos_length])
		self.uLocs_seq = tf.placeholder(name='uLocs_seq', dtype=tf.int32, shape=[None])
		self.suids=list()
		self.siids=list()
		self.suLocs_seq=list()
		for k in range(args.graphNum):
			self.suids.append(tf.placeholder(name='suids%d'%k, dtype=tf.int32, shape=[None]))
			self.siids.append(tf.placeholder(name='siids%d'%k, dtype=tf.int32, shape=[None]))
			self.suLocs_seq.append(tf.placeholder(name='suLocs%d'%k, dtype=tf.int32, shape=[None]))
		self.subAdj=list()
		self.subTpAdj=list()
		# self.subAdjNp=list()
		for i in range(args.graphNum):
			seqadj = self.handler.subMat[i]
			idx, data, shape = transToLsts(seqadj, norm=True)
			print("1",shape)
			self.subAdj.append(tf.sparse.SparseTensor(idx, data, shape))
			idx, data, shape = transToLsts(transpose(seqadj), norm=True)
			self.subTpAdj.append(tf.sparse.SparseTensor(idx, data, shape))
			print("2",shape)
		self.maxTime=self.handler.maxTime
		#############################################################################
		self.preds, self.sslloss = self.ours()
		sampNum = tf.shape(self.uids)[0] // 2
		self.posPred = tf.slice(self.preds, [0], [sampNum])# begin at 0, size = sampleNum
		self.negPred = tf.slice(self.preds, [sampNum], [-1])# 
		self.preLoss = tf.reduce_mean(tf.maximum(0.0, 1.0 - (self.posPred - self.negPred)))# +tf.reduce_mean(tf.maximum(0.0,self.negPred))
		self.regLoss = args.reg * Regularize()  + args.ssl_reg * self.sslloss
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)
	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Initiated')
		maxndcg=0.0
		maxres=dict()
		maxepoch=0
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.tstEpoch == 0 and reses['NDCG']>maxndcg:
				self.saveHistory()
				maxndcg=reses['NDCG']
				maxres=reses
				maxepoch=ep
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		log(self.makePrint('max', maxepoch, maxres, True))