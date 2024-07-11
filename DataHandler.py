import os
import pickle
import numpy as np
import scipy.sparse as sp
from Params import args
from scipy.sparse import csr_matrix


class Datahandler:
    def __init__(self):
        if args.data == 'amazon':
            self.predict_directory = './Datasets/amazon/'
        else:
            self.predict_directory = f'./Datasets/{args.data}/'
        self.trainfile = self.predict_directory + 'trn_mat_time'
        self.testfile = self.predict_directory + 'tst_int'
        self.sequencefile = self.predict_directory + 'sequence'
        self.dictfile = self.predict_directory + 'test_dict'
    
    def LoadData(self):
        if args.percent > 1e-8:
            print('noised')
            with open(self.predict_directory + 'noise_%.2f' % args.percent, 'rb') as fs:
                trnMat = pickle.load(fs)
        else:
            with open(self.trainfile,'rb') as fs:
                trnMat = pickle.load(fs)
        if os.path.isfile(self.testfile):
            with open(self.testfile, 'rb') as fs:
                tstInt = pickle.load(fs)
        if os.path.isfile(self.sequencefile):
            with open(self.sequencefile, 'rb') as fs:
                seqfile = pickle.load(fs)
        if os.path.isfile(self.dictfile):
            with open(self.dictfile, 'rb') as fs:
                dictfile = pickle.load(fs)
        #print("tstInt",tstInt)
        tstStat = (np.array(tstInt) != None)
        #print("tstStat",tstStat,len(tstStat))
        tstUsrs = np.reshape(np.argwhere(tstStat != False), [-1])
        #print("tstUsrs",tstUsrs,len(tstUsrs))

        def generate_rating_matrix_test(user_seq, num_users, num_items):
			# three lists are used to construct sparse matrix
            row = []
            col = []
            data = []
            for user_id, item_list in enumerate(user_seq):
                for item in item_list:
                    row.append(user_id)
                    col.append(item)
                    data.append(1)

            row = np.array(row)
            col = np.array(col)
            data = np.array(data)
            rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

            return rating_matrix
        
        args.user, args.item = trnMat[0].shape
        self.trnMat = generate_rating_matrix_test(seqfile, args.user, args.item)
        self.subMat = trnMat[1]
        self.timeMat = trnMat[2]
        #print("trnMat",trnMat[0],trnMat[1],trnMat[2])
        self.tstInt = tstInt
        self.tstUsrs = tstUsrs
        self.prepareGlobalData()

    def prepareGlobalData(self):
        def tran_to_sym(R):
            adj_mat = sp.dok_matrix((args.user + args.item, args.user + args.item), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = R.tolil()
            adj_mat[:args.user, args.user:] = R
            adj_mat[args.user:, :args.user] = R.T
            adj_mat = adj_mat.tocsr()
            return (adj_mat+sp.eye(adj_mat.shape[0]))
			

		# adj = self.subMat
        self.maxTime=1
		# self.subMat,self.maxTime=self.timeProcess(self.subMat)
        #print(self.subMat[0],self.subMat[-1])
    
    def timeProcess(self,trnMats):
        mi = 1e16
        ma = 0
        for i in range(len(trnMats)):
            minn = np.min(trnMats[i].data)
            maxx = np.max(trnMats[i].data)
            mi = min(mi, minn)
            ma = max(ma, maxx)
        maxTime = 0
        for i in range(len(trnMats)):
            newData = ((trnMats[i].data - mi) // (3600*24*args.slot)).astype(np.int32)
            maxTime = max(np.max(newData), maxTime)
            trnMats[i] = csr_matrix((newData, trnMats[i].indices, trnMats[i].indptr), shape=trnMats[i].shape)
        print('MAX TIME',mi,ma, maxTime)
        return trnMats, maxTime + 1
    
if __name__ == '__main__':
    Datahandler().LoadData()