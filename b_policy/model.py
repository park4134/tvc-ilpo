from tensorflow.keras.layers import Dense, LSTM, Conv1D, LeakyReLU, Softmax
from tensorflow.keras import Model, Input
import tensorflow as tf

class StateEmbedding(tf.keras.layers.Layer):
    def __init__(self, units, layer_num, lrelu=0.2):
        super(StateEmbedding, self).__init__()
        self.units = units
        self.layer_num = layer_num
        self.lrelu = lrelu

        self.s_e_dense = []
        self.s_e_lrelu = []

        self.s_e_dense.append(Dense(units=self.units, name = 'State_Embedding_Dense_0'))

        for i in range(self.layer_num):
            self.s_e_lrelu.append(LeakyReLU(alpha=self.lrelu, name = f'State_Embedding_lrelu_{i}'))
            self.s_e_dense.append(Dense(units=2 * self.units, name = f'State_Embedding_Dense_{i+1}'))
    
    def call(self, inputs):
        s_e = self.s_e_dense[0](inputs)
        for i in range(self.layer_num):
            s_e = self.s_e_lrelu[i](s_e)
            s_e = self.s_e_dense[i+1](s_e)

        return s_e
    
class SeqStateEmbedding(tf.keras.layers.Layer):
    def __init__(self, units, layer_num, lrelu=0.2):
        super(SeqStateEmbedding, self).__init__()
        self.units = units
        self.layer_num = layer_num
        self.lrelu = lrelu

        self.s_e_lstm = []
        self.s_e_lrelu = []

        for i in range(self.layer_num):
            self.s_e_lstm.append(LSTM(units=self.units, return_sequences=True, name = f'State_Embedding_LSTM_{i}'))
            self.s_e_lrelu.append(LeakyReLU(alpha=self.lrelu, name = f'State_Embedding_lrelu_{i}'))

        self.s_e_lstm_top = LSTM(units=2 * self.units, return_sequences=False, name = f'State_Embedding_LSTM_{self.layer_num}')
        self.s_e_dense_top = Dense(units=2 * self.units, name='State_Embedding_Dense_top')
    
    def call(self, inputs):
        s_e = inputs
        for i in range(self.layer_num):
            s_e = self.s_e_lstm[i](s_e)
            s_e = self.s_e_lrelu[i](s_e)
        
        s_e = self.s_e_lstm_top(s_e)
        s_e = self.s_e_dense_top(s_e)

        return s_e

class PolicyNetwork(Model):
    def __init__(self, n_state, n_latent_action, units, layer_num, batch_size, lrelu=0.2):
        super().__init__()
        self.n_state = n_state
        self.n_latent_action = n_latent_action
        self.units = units
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.lrelu = lrelu

        self.state_embedding = StateEmbedding(
            units = self.units,
            layer_num = self.layer_num,
            lrelu = self.lrelu
        )
        self.state_embedding_lrelu_top = LeakyReLU(alpha=self.lrelu, name = 'State_Embedding_lrelu_top')
        self.state_embedding_dense_top = Dense(units=self.n_latent_action, name = 'State_Embedding_Dense_top')
        self.state_embedding_softmax = Softmax(name='Action_Probability_Softmax')

        self.action_embedding_dense = Dense(units = 2*self.units, name = 'Action_Embedding_Dense')
        self.action_embedding_lrelu = LeakyReLU(alpha = self.lrelu, name = 'Action_Embedding_lrelu')

        self.generator_dense = []
        self.generator_dense.append(Dense(units = 2*self.units, name = 'Generator_Dense_0'))
        self.generator_dense.append(Dense(units = self.units, name = 'Generator_Dense_1'))
        self.generator_dense.append(Dense(units = self.n_state, name = 'Generator_Dense_top'))

        self.generator_lrelu = []
        self.generator_lrelu.append(LeakyReLU(alpha=self.lrelu, name = 'Generator_lrelu_0'))
        self.generator_lrelu.append(LeakyReLU(alpha=self.lrelu, name = 'Generator_lrelu_1'))

    def build_graph(self):
        self.input_layer = Input(shape=(self.n_state,), name='Input_layer')
        self.out = self.call(self.input_layer)
        self.build(input_shape=(self.batch_size, self.n_state))
        self.summary()

        return Model(inputs=[self.input_layer], outputs=self.out)
    
    def call(self, inputs, training=False):
        s_e = self.state_embedding(inputs) # e : (batch, 2*units), z_p : (batch, n_latent_action)
        e = self.state_embedding_lrelu_top(s_e)
        z_p = self.state_embedding_dense_top(e)
        z_p = self.state_embedding_softmax(z_p)

        for latent_act in range(self.n_latent_action):
            z_onehot = tf.one_hot([latent_act], self.n_latent_action)
            z = self.action_embedding_dense(z_onehot)
            z = self.action_embedding_lrelu(z) # z : (1, 2*units)

            z = tf.tile(z, [1, tf.shape(e)[0]])
            z = tf.reshape(z, (tf.shape(e)[0], 2*self.units)) # z : (batch, 2*units)

            concat = tf.concat([e, z], axis=-1) # concat : (batch, 4*units)

            for i in range(len(self.generator_lrelu)):
                concat = self.generator_dense[i](concat)
                concat = self.generator_lrelu[i](concat)

            g = self.generator_dense[-1](concat) # g : (batch, n_state)
            g = tf.expand_dims(g, axis=1) # g : (batch, 1, n_state)

            if latent_act == 0:
                delta_s = g
            else:
                delta_s = tf.concat([delta_s, g], axis=1) # delta_s : (batch, n_latent_action, n_state)
        
        return z_p, delta_s
    
class SeqPolicyNetwork(PolicyNetwork):
    def __init__(self, n_state, n_latent_action, seq, units, layer_num, batch_size, lrelu=0.2):
        super().__init__(
            n_state = n_state,
            n_latent_action = n_latent_action,
            units = units,
            layer_num = layer_num,
            batch_size = batch_size
        )
        self.seq = seq

        self.state_embedding = SeqStateEmbedding(
            units = self.units,
            layer_num = self.layer_num,
            lrelu = self.lrelu
        )
    
    def build_graph(self):
        self.input_layer = Input(shape=(self.seq, self.n_state,), name='Input_layer')
        self.out = self.call(self.input_layer)
        self.build(input_shape=(self.batch_size, self.seq, self.n_state))
        self.summary()

        return Model(inputs=[self.input_layer], outputs=self.out)

if __name__ == '__main__':
    # model = PolicyNetwork(
    #     n_state=6,
    #     n_latent_action=3,
    #     units=64,
    #     layer_num=1,
    #     batch_size=128
    # )
    model = SeqPolicyNetwork(
        n_state=6,
        n_latent_action=3,
        seq=10,
        units=64,
        layer_num=1,
        batch_size=128
    )

    model.build_graph()