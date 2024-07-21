import warnings
import os
import gc

# Suppress all warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from ast import arg
from curses import meta
from site import USER_BASE
from matplotlib.cbook import silent_list
from Params import args
import utils.NNLayers as NNs
from utils.NNLayers import Regularize
import tensorflow as tf
import pickle
import utils.TimeLogger as logger
from utils.util import *
import numpy as np
from utils.TimeLogger import log
from DataHandler import negSamp, transpose, transToLsts
import scipy.sparse as sp
from random import randint
from model import Model

class Recommender:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'HR', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

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

    def run(self):
        self.prepareModel()
        if args.load_model is not None:
            self.loadModel(args.load_model)
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            # Variables are initialized automatically in TensorFlow 2.x
            print('Variables Initialized')

        maxndcg = 0.0
        maxres = {}
        maxepoch = 0

        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            print(self.makePrint('Train', ep, reses, test))
            
            if test:
                reses = self.testEpoch()
                print(self.makePrint('Test', ep, reses, test))

            if ep % args.tstEpoch == 0 and reses['NDCG'] > maxndcg:
                self.saveHistory()
                maxndcg = reses['NDCG']
                maxres = reses
                maxepoch = ep

            print()

        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        log(self.makePrint('max', maxepoch, maxres, True))

    def loss_fn(self, uids, preds, sslloss):
        sampNum = tf.shape(uids)[0] // 2
        posPred = tf.slice(preds, [0], [sampNum])
        negPred = tf.slice(preds, [sampNum], [-1])
        preLoss = tf.reduce_mean(tf.maximum(0.0, 1.0 - (posPred - negPred)))
        regLoss = args.reg * self.compute_reg() + args.ssl_reg * sslloss
        loss = preLoss + regLoss
        return loss, preLoss, regLoss

    
    def prepareModel(self):
        NNs.leaky = args.leaky
        
        subAdj = []
        subTpAdj = []

        for i in range(args.graphNum):
            seqadj = self.handler.subMat[i]
            idx, data, shape = transToLsts(seqadj, norm=True)
            subAdj.append(tf.sparse.SparseTensor(idx, data, shape))
            idx, data, shape = transToLsts(transpose(seqadj), norm=True)
            subTpAdj.append(tf.sparse.SparseTensor(idx, data, shape))

        
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
        learningRate = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learningRate)
        self.maxTime = self.handler.maxTime

        self.model = Model(subAdj= subAdj, subTpAdj=subTpAdj)
        




    def sampleTrainBatch(self, batIds, labelMat, timeMat, train_sample_num):
        temTst = self.handler.tstInt[batIds]
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * train_sample_num
        
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        uLocs_seq = [None]* temlen
        sequence = [None] * args.batch
        mask = [None]*args.batch
        cur = 0	

        for i in range(batch):
            posset = self.handler.sequence[batIds[i]][:-1]
            sampNum = min(train_sample_num, len(posset))
            choose = 1

            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = []
                choose = 1
                poslocs.extend([posset[-choose]] * sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item, [self.handler.sequence[batIds[i]][-1], temTst[i]], self.handler.item_with_pop)

            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
                uLocs_seq[cur] = uLocs_seq[cur+temlen//2] = i
                iLocs[cur] = posloc
                iLocs[cur+temlen//2] = negloc
                cur += 1

            sequence[i]=np.zeros(args.pos_length,dtype=int)
            mask[i]=np.zeros(args.pos_length)
            posset=posset[:-choose]

            if(len(posset)<=args.pos_length):
                sequence[i][-len(posset):]=posset
                mask[i][-len(posset):]=1
            else:
                sequence[i]=posset[-args.pos_length:]
                mask[i]+=1

        uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
        uLocs_seq = uLocs_seq[:cur] + uLocs_seq[temlen//2: temlen//2 + cur]
        if(batch<args.batch):
            for i in range(batch,args.batch):
                sequence[i]=np.zeros(args.pos_length,dtype=int)
                mask[i]=np.zeros(args.pos_length)
        return uLocs, iLocs, sequence,mask, uLocs_seq
    
    def sampleSslBatch(self, batIds, labelMat, use_epsilon=True):
        temLabel = [labelMat[k][batIds].toarray() for k in range(args.graphNum)]
        batch = len(batIds)
        temlen = batch * 2 * args.sslNum
        
        uLocs = [[None] * temlen] * args.graphNum
        iLocs = [[None] * temlen] * args.graphNum
        uLocs_seq = [[None] * temlen] * args.graphNum

        for k in range(args.graphNum):
            cur = 0

            for i in range(batch):
                posset = np.reshape(np.argwhere(temLabel[k][i] != 0), [-1])
                sslNum = min(args.sslNum, len(posset) // 2)

                if sslNum == 0:
                    poslocs = [np.random.choice(args.item)]
                    neglocs = [poslocs[0]]
                else:
                    all_samples = np.random.choice(posset, sslNum * 2)
                    poslocs = all_samples[:sslNum]
                    neglocs = all_samples[sslNum:]

                for j in range(sslNum):
                    posloc = poslocs[j]
                    negloc = neglocs[j]
                    uLocs[k][cur] = uLocs[k][cur+1] = batIds[i]
                    uLocs_seq[k][cur] = uLocs_seq[k][cur+1] = i
                    iLocs[k][cur] = posloc
                    iLocs[k][cur+1] = negloc
                    cur += 2

            uLocs[k] = uLocs[k][:cur]
            iLocs[k] = iLocs[k][:cur]
            uLocs_seq[k] = uLocs_seq[k][:cur]

        return uLocs, iLocs, uLocs_seq
    
    def print_trainable_parameters(self, model):
        total_parameters = 0
        for layer in model.layers:
            for variable in layer.trainable_variables:
                shape = variable.shape
                variable_parameters = tf.reduce_prod(shape).numpy()
                print(f'Variable: {variable.name}, Shape: {shape}, Total parameters: {variable_parameters}')
                total_parameters += variable_parameters
        print(f'Total trainable parameters: {total_parameters}')
    
    def get_all_regularized_variables(self):
        regularized_vars = []

        # Check model-level weights
        for var in self.model.weights:
            if hasattr(var, 'regularizer') and var.regularizer is not None:
                regularized_vars.append(var)

        # Check layer-level weights
        for layer in self.model.layers:
            for var in layer.weights:
                if hasattr(var, 'regularizer') and var.regularizer is not None:
                    regularized_vars.append(var)

        return regularized_vars
    

    def compute_reg(self):
        l2_loss = 0.0
        for var in self.get_all_regularized_variables():
            l2_loss += tf.reduce_sum(tf.square(var))
        return l2_loss
    
    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            pred, ssloss = self.model(inputs, training=True)
            loss, preLoss, regLoss = self.loss_fn(inputs['uids'], pred, ssloss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.global_step.assign_add(1)

        return loss, preLoss, regLoss
    
    def reset_memory(self):
        tf.keras.backend.clear_session()
        gc.collect()

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        num = len(sfIds)
        sample_num_list = [40]
        steps = int(np.ceil(num / args.batch))
        
        for s in range(len(sample_num_list)):
            for i in range(steps):
                st = i * args.batch
                ed = min((i + 1) * args.batch, num)
                batIds = sfIds[st: ed]

                uLocs, iLocs, sequence, mask, uLocs_seq = self.sampleTrainBatch(batIds, self.handler.trnMat, self.handler.timeMat, sample_num_list[s])
                suLocs, siLocs, _ = self.sampleSslBatch(batIds, self.handler.subMat, False)

                inputs = {
                        'uids': tf.constant(uLocs),
                        'iids': tf.constant(iLocs),
                        'sequence': tf.constant(sequence, shape = [args.batch, args.pos_length]),
                        'mask': tf.constant(mask, shape = [args.batch, args.pos_length]),
                        'uLocs_seq': tf.constant(uLocs_seq),
                        **{f'suids{k}': tf.constant(suLocs[k]) for k in range(args.graphNum)},
                        **{f'siids{k}': tf.constant(siLocs[k]) for k in range(args.graphNum)}
                        }

                loss, preLoss, regLoss = self.train_step(inputs)

                epochLoss += loss
                epochPreLoss += preLoss
                log('Step %d/%d: preloss = %.2f, REGLoss = %.2f         ' % (i+s*steps, steps*len(sample_num_list), preLoss, regLoss), save=False, oneline=True)

        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        return ret
    
    def sampleTestBatch(self, batIds, labelMat): # labelMat=TrainMat(adj)
        batch = len(batIds)
        temTst = self.handler.tstInt[batIds]
        temLabel = labelMat[batIds].toarray()
        temlen = batch * args.testSize# args.item
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        uLocs_seq = [None] * temlen
        tstLocs = [None] * batch
        sequence = [None] * args.batch
        mask = [None]*args.batch
        cur = 0
        val_list=[None]*args.batch

        for i in range(batch):
            if(args.test==True):
                posloc = temTst[i]
            else:
                posloc = self.handler.sequence[batIds[i]][-1]
                val_list[i]=posloc

            rdnNegSet = np.array(self.handler.test_dict[batIds[i]+1][:args.testSize-1])-1
            locset = np.concatenate((rdnNegSet, np.array([posloc])))
            tstLocs[i] = locset

            for j in range(len(locset)):
                uLocs[cur] = batIds[i]
                iLocs[cur] = locset[j]
                uLocs_seq[cur] = i
                cur += 1

            sequence[i]=np.zeros(args.pos_length,dtype=int)
            mask[i]=np.zeros(args.pos_length)

            if(args.test==True):
                posset=self.handler.sequence[batIds[i]]
            else:
                posset=self.handler.sequence[batIds[i]][:-1]
			# posset=self.handler.sequence[batIds[i]]
                  
            if(len(posset)<=args.pos_length):
                sequence[i][-len(posset):]=posset
                mask[i][-len(posset):]=1
            else:
                sequence[i]=posset[-args.pos_length:]
                mask[i]+=1

        if(batch<args.batch):
            for i in range(batch,args.batch):
                sequence[i]=np.zeros(args.pos_length,dtype=int)
                mask[i]=np.zeros(args.pos_length)

        return uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list
    
    def testEpoch(self):
        epochHit, epochNdcg = [0] * 2
        epochHit5, epochNdcg5 = [0] * 2
        epochHit20, epochNdcg20 = [0] * 2
        epochHit1, epochNdcg1 = [0] * 2
        epochHit15, epochNdcg15 = [0] * 2
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
          
        for i in range(steps):
            st = i * tstBat
            ed = min((i+1) * tstBat, num)
            batIds = ids[st: ed]
            feed_dict = {}
            uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list = self.sampleTestBatch(batIds, self.handler.trnMat)
            suLocs, siLocs, _ = self.sampleSslBatch(batIds, self.handler.subMat, False)

            inputs = {
                        'uids': tf.constant(uLocs),
                        'iids': tf.constant(iLocs),
                        'sequence': tf.constant(sequence, shape = [args.batch, args.pos_length]),
                        'mask': tf.constant(mask, shape = [args.batch, args.pos_length]),
                        'uLocs_seq': tf.constant(uLocs_seq),
                        **{f'suids{k}': tf.constant(suLocs[k]) for k in range(args.graphNum)},
                        **{f'siids{k}': tf.constant(siLocs[k]) for k in range(args.graphNum)}
                        }

            preds, ssloss = self.model(inputs)

            if(args.uid!=-1):
                print(preds[args.uid].numpy())
            if(args.test==True):
                hit, ndcg, hit5, ndcg5, hit20, ndcg20,hit1, ndcg1,  hit15, ndcg15= self.calcRes(np.reshape(preds, [ed-st, args.testSize]), temTst, tstLocs)
            else:
                hit, ndcg, hit5, ndcg5, hit20, ndcg20,hit1, ndcg1,  hit15, ndcg15= self.calcRes(np.reshape(preds, [ed-st, args.testSize]), val_list, tstLocs)

            epochHit += hit
            epochNdcg += ndcg
            epochHit5 += hit5
            epochNdcg5 += ndcg5
            epochHit20 += hit20
            epochNdcg20 += ndcg20
            epochHit15 += hit15
            epochNdcg15 += ndcg15
            epochHit1 += hit1
            epochNdcg1 += ndcg1

            log('Steps %d/%d: hit10 = %d, ndcg10 = %d' % (i, steps, hit, ndcg), save=False, oneline=True)

        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        print("epochNdcg1:{},epochHit1:{},epochNdcg5:{},epochHit5:{}".format(epochNdcg1/ num,epochHit1/ num,epochNdcg5/ num,epochHit5/ num))
        print("epochNdcg15:{},epochHit15:{},epochNdcg20:{},epochHit20:{}".format(epochNdcg15/ num,epochHit15/ num,epochNdcg20/ num,epochHit20/ num))
        return ret
    
    def calcRes(self, preds, temTst, tstLocs):
        hit = 0
        ndcg = 0
        hit1 = 0
        ndcg1 = 0
        hit5=0
        ndcg5=0
        hit20=0
        ndcg20=0
        hit15=0
        ndcg15=0

        for j in range(preds.shape[0]):
            predvals = list(zip(preds[j], tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:args.shoot]))

            if temTst[j] in shoot:
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
            shoot = list(map(lambda x: x[1], predvals[:5]))

            if temTst[j] in shoot:
                hit5 += 1
                ndcg5 += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
            shoot = list(map(lambda x: x[1], predvals[:20]))

            if temTst[j] in shoot:
                hit20 += 1
                ndcg20 += np.reciprocal(np.log2(shoot.index(temTst[j])+2))	
        return hit, ndcg, hit5, ndcg5, hit20, ndcg20, hit1, ndcg1, hit15, ndcg15
    
    def saveHistory(self):
        if args.epoch == 0:
            return
        
        # Save training history
        with open('History/' + args.save_path + '.pkl', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        # Save model
        model_save_path = 'Models/' + args.save_path + '.keras'
        tf.keras.models.save_model(self.model, model_save_path)
        
        log('Model Saved: %s' % model_save_path)
        
    def loadModel(self):
        # Load the model
        model_path = 'Models/' + args.load_model + '.keras'
        if os.path.exists(model_path):
            loaded_model = tf.keras.models.load_model(model_path)
            log('Model Loaded')
        else:
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        # Load the history
        history_path = 'History/' + args.load_model + '.pkl'
        if os.path.exists(history_path):
            with open(history_path, 'rb') as fs:
                self.metrics = pickle.load(fs)
        else:
            self.metrics = {}
            log(f"History file '{history_path}' not found. Metrics initialized as empty.")
