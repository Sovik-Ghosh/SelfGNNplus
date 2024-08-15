import tensorflow as tf
from Params import args
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout
from utils.NNLayers import GRUNet, LSTMNet, TemporalConvNet, TransformerNet, MultiHeadSelfAttention, AdditiveAttention


class Model(tf.keras.Model):
    def __init__(self, subAdj, subTpAdj, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.subAdj = subAdj
        self.subTpAdj = subTpAdj
        self.maxTime = 1
        self.actFunc = 'leakyRelu'
        self.users = tf.range(args.user)
        self.items = tf.range(args.item)

        self.keep_rate = args.keepRate
        self.dropout = 1 - self.keep_rate
        self.gnn_layers = args.gnn_layer
        self.att_layers = args.att_layer
        self.latdim = args.latdim
        self.num_heads = args.num_attention_heads
        self.query_vector_dim = args.query_vector_dim
        self.layer_norma0 = LayerNormalization()
        self.layer_norma1 = LayerNormalization()
        self.layer_norma2 = LayerNormalization()
        self.layer_norma3 = LayerNormalization()
        self.layer_norma4 = LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(args.ssldim, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(args.reg))
        self.fc2 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(args.reg))
        

        self.gru0 = GRUNet(hidden_units=self.latdim, dropout = 1 - self.keep_rate)
        self.gru1 = GRUNet(hidden_units=self.latdim, dropout = 1 - self.keep_rate)
        self.lstm0 = LSTMNet(hidden_units=self.latdim, dropout = 1 - self.keep_rate)
        self.lstm1 = LSTMNet(hidden_units=self.latdim, dropout = 1 - self.keep_rate)
        self.tcn0 = TemporalConvNet([self.latdim, self.latdim], stride=1, kernel_size=3, dropout=self.dropout)
        self.tcn1 = TemporalConvNet([self.latdim, self.latdim], stride=1, kernel_size=3, dropout=self.dropout)
        self.tnet0 = TransformerNet(num_units=args.latdim, num_blocks=3, num_heads=self.num_heads, maxlen=args.pos_length, dropout_rate=self.dropout, pos_fixed=False, l2_reg=args.reg)
        self.tnet1 = TransformerNet(num_units=args.latdim, num_blocks=3, num_heads=self.num_heads, maxlen=args.pos_length, dropout_rate=self.dropout, pos_fixed=False, l2_reg=args.reg)
        self.additive_attention0 = AdditiveAttention()
        self.additive_attention1 = AdditiveAttention()
        self.multihead_self_attention0 = MultiHeadSelfAttention(self.latdim, self.num_heads)
        self.multihead_self_attention1 = MultiHeadSelfAttention(self.latdim, self.num_heads)
        self.multihead_self_attention_sequence = [MultiHeadSelfAttention(self.latdim, self.num_heads) for _ in range(args.att_layer)]
        
    def build(self, input_shape):
        self.position_embedding = self.add_weight(
            name='position_embedding',
            shape=[args.pos_length, args.latdim],
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(args.reg),
            trainable=True
        )
        self.user_embeddings = self.add_weight(
            name='user_embeddings',
            shape=[args.graphNum, args.user, args.latdim],
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(args.reg),
            trainable=True
        )
        self.item_embeddings = self.add_weight(
            name='item_embeddings',
            shape=[args.graphNum, args.item, args.latdim],
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(args.reg),
            trainable=True
        )
        super(Model, self).build(input_shape)

    def message_propagate(self, srclats, mat, type='user', training = False):
        srcNodes = tf.squeeze(tf.slice(mat.indices, [0, 1], [-1, 1]))
        tgtNodes = tf.squeeze(tf.slice(mat.indices, [0, 0], [-1, 1]))
        srcEmbeds = tf.nn.embedding_lookup(srclats, srcNodes)
        
        lat = tf.pad(tf.math.segment_sum(srcEmbeds, tgtNodes), [[0, 100], [0, 0]])

        if type == 'user':
            lat = tf.nn.embedding_lookup(lat, self.users)
        else:
            lat = tf.nn.embedding_lookup(lat, self.items)

        return self._activate(lat)

    def edge_dropout(self, mat, training):
        def dropOneMat(mat, training):
            indices = mat.indices
            values = mat.values
            newVals = tf.nn.dropout(tf.cast(values, dtype=tf.float32), rate=self.dropout if training else 0.0)
            return tf.sparse.SparseTensor(indices, tf.cast(newVals, dtype=tf.int32), mat.dense_shape)
        return dropOneMat(mat, training)

    def _activate(self, lat):
        if self.actFunc == 'leakyRelu':
            return tf.maximum(lat, args.leaky * lat)
        return lat
    
    def call(self, inputs, training=False):

        user_vectors, item_vectors = [], []
        pos = tf.tile(tf.expand_dims(tf.range(args.pos_length), axis=0), [args.batch, 1])
        for k in range(args.graphNum):
            embs0, embs1 = [self.user_embeddings[k]], [self.item_embeddings[k]]
            for _ in range(self.gnn_layers):
                a_emb0 = self.message_propagate(embs1[-1], self.edge_dropout(self.subAdj[k], training), 'user', training = training)
                a_emb1 = self.message_propagate(embs0[-1], self.edge_dropout(self.subTpAdj[k], training), 'item', training = training)
                embs0.append(a_emb0 + embs0[-1])
                embs1.append(a_emb1 + embs1[-1])
                
            user_vectors.append(tf.add_n(embs0))
            item_vectors.append(tf.add_n(embs1))

        user_vector = tf.stack(user_vectors, axis=0)
        item_vector = tf.stack(item_vectors, axis=0)
        
        user_vector_tensor = tf.transpose(user_vector, perm=[1, 0, 2])
        item_vector_tensor = tf.transpose(item_vector, perm=[1, 0, 2])

        if args.model == 'gru':
            user_vector_tensor = self.gru0(inputs=user_vector_tensor, training = training)
            item_vector_tensor = self.gru1(inputs=item_vector_tensor, training = training)
        elif args.model == 'lstm':
            user_vector_tensor = self.lstm0(inputs=user_vector_tensor, training = training)
            item_vector_tensor = self.lstm1(inputs=item_vector_tensor, training = training)
        elif args.model == 'tcn':
            user_vector_tensor = self.tcn0(inputs=user_vector_tensor, training = training)
            item_vector_tensor = self.tcn1(inputs=item_vector_tensor, training = training)
        elif args.model == 'tnet':
            user_vector_tensor = self.tnet0(inputs=user_vector_tensor, training = training)
            item_vector_tensor = self.tnet1(inputs=item_vector_tensor, training = training)
        else:
            pass
        
        if args.long_use:
            multihead_user_vector = self.multihead_self_attention0(self.layer_norma0(user_vector_tensor))
            multihead_item_vector = self.multihead_self_attention1(self.layer_norma1(item_vector_tensor))
            
            final_user_vector = tf.reduce_mean(multihead_user_vector, axis=1)
            final_item_vector = tf.reduce_mean(multihead_item_vector, axis=1)
            
            sequence_batch = self.layer_norma2(tf.matmul(tf.expand_dims(inputs['mask'], axis=1), tf.nn.embedding_lookup(final_item_vector, tf.cast(inputs['sequence'], tf.int32))))
            sequence_batch += self.layer_norma3(tf.matmul(tf.expand_dims(inputs['mask'], axis=1), tf.nn.embedding_lookup(self.position_embedding, pos)))
            att_layer = sequence_batch
            
            for i in range(self.att_layers):
                att_layer1 = self.multihead_self_attention_sequence[i](self.layer_norma4(att_layer))
                att_layer = self._activate(att_layer1) + att_layer
            
            att_user = tf.reduce_sum(att_layer, axis=1)

            pckUlat = tf.nn.embedding_lookup(final_user_vector, tf.cast(inputs['uids'], tf.int32))
            pckIlat = tf.nn.embedding_lookup(final_item_vector, tf.cast(inputs['iids'], tf.int32))
            
            preds = tf.reduce_sum(pckUlat * pckIlat, axis=-1)
            preds += tf.reduce_sum(self._activate(tf.nn.embedding_lookup(att_user, tf.cast(inputs['uLocs_seq'], tf.int32))) * pckIlat, axis=-1)
        else:
            final_user_vector = tf.reduce_mean(user_vector_tensor, axis=1)
            final_item_vector = tf.reduce_mean(item_vector_tensor, axis=1)
            
            pckUlat = tf.nn.embedding_lookup(final_user_vector, tf.cast(inputs['uids'], tf.int32))
            pckIlat = tf.nn.embedding_lookup(final_item_vector, tf.cast(inputs['iids'], tf.int32))
            
            preds = tf.reduce_sum(pckUlat * pckIlat, axis=-1)
            preds += tf.reduce_sum(self._activate(tf.nn.embedding_lookup(tf.reduce_sum(user_vector_tensor, axis=1), tf.cast(inputs['uLocs_seq'], tf.int32))) * pckIlat, axis=-1)
            
        user_weight = []
        preds_one = []
        sslloss = 0
        
        if args.ssl:
            for i in range(args.graphNum):
                meta1 = tf.concat([final_user_vector * user_vector[i], final_user_vector, user_vector[i]], axis=-1)
                meta2 = self.fc1(meta1)
                meta3 = self.fc2(meta2)
                user_weight.append(tf.squeeze(meta3))
            user_weight = tf.stack(user_weight, axis=0)

            
            
            for i in range(args.graphNum):
                sampNum = tf.shape(inputs[f'suids{i}'])[0] // 2
                pckUlat = tf.nn.embedding_lookup(final_user_vector, tf.cast(inputs[f'suids{i}'], tf.int32))
                pckIlat = tf.nn.embedding_lookup(final_item_vector, tf.cast(inputs[f'siids{i}'], tf.int32))
                pckUweight = tf.nn.embedding_lookup(user_weight[i], tf.cast(inputs[f'suids{i}'], tf.int32))
                
                S_final = tf.reduce_sum(self._activate(pckUlat * pckIlat), axis=-1)
                posPred_final = tf.stop_gradient(tf.slice(S_final, [0], [sampNum]))
                negPred_final = tf.stop_gradient(tf.slice(S_final, [sampNum], [-1]))
                posweight_final = tf.slice(pckUweight, [0], [sampNum])
                negweight_final = tf.slice(pckUweight, [sampNum], [-1])
                S_final = posweight_final * posPred_final - negweight_final * negPred_final
                
                pckUlat = tf.nn.embedding_lookup(user_vector[i], tf.cast(inputs[f'suids{i}'], tf.int32))
                pckIlat = tf.nn.embedding_lookup(item_vector[i], tf.cast(inputs[f'siids{i}'], tf.int32))
                preds_one = tf.reduce_sum(self._activate(pckUlat * pckIlat), axis=-1)
                posPred = tf.slice(preds_one, [0], [sampNum])
                negPred = tf.slice(preds_one, [sampNum], [-1])
                sslloss += tf.reduce_sum(tf.maximum(0.0, 1.0 - S_final * (posPred - negPred)))
        
        return preds, sslloss