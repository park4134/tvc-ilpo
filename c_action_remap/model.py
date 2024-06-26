import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tensorflow.keras.layers import Dense, LeakyReLU, Softmax
from tensorflow.keras import Model, Input
from b_policy.model import StateEmbedding
import tensorflow as tf
import yaml

class ActionRemapNetwork(Model):
    def __init__(self, n_state, n_latent_action, n_action, units, units_p, layer_num, layer_num_p, batch_size, lrelu=0.2):
        super().__init__()
        self.n_state = n_state
        self.n_latent_action = n_latent_action
        self.n_action = n_action
        self.units = units
        self.units_p = units_p
        self.layer_num = layer_num
        self.layer_num_p = layer_num_p
        self.batch_size = batch_size
        self.lrelu = lrelu

        '''State Embedding'''
        self.state_embedding = StateEmbedding(
            units=self.units_p,
            layer_num=self.layer_num_p,
            lrelu = self.lrelu
        )
        self.state_embedding_lrelu = LeakyReLU(alpha=self.lrelu, name='State_Embedding_lrelu')

        '''Action Embedding'''
        self.action_embedding_dense = Dense(units=2*self.units_p, name = 'Action_Embedding_Dense')
        self.action_embedding_lrelu = LeakyReLU(alpha = self.lrelu, name = 'Action_Embedding_lrelu')

        '''Action Remapping'''
        self.action_remapping_dense = []
        self.action_remapping_lrelu = []
        for i in range(self.layer_num):
            self.action_remapping_dense.append(Dense(units=2*self.units, name = f'Action_Remap_Dense_{i}'))
            self.action_remapping_lrelu.append(LeakyReLU(alpha = self.lrelu, name = f'Action_Remap_lrelu_{i}'))
        
        self.action_remapping_dense.append(Dense(units=self.units, name = f'Action_Remap_Dense_{self.layer_num}'))
        self.action_remapping_lrelu.append(LeakyReLU(alpha = self.lrelu, name = f'Action_Remap_lrelu_{self.layer_num}'))

        self.action_remapping_dense_top = (Dense(units=self.n_action, name = 'Action_Remap_Dense_top'))
        self.action_remapping_softmax = (Softmax(name = 'Action_Remap_softmax'))

    def build_graph(self):
        self.input_layer1 = Input(shape=(self.n_state,), name='input_layer1')
        self.input_layer2 = Input(shape=(self.n_latent_action,), name='input_layer2')
        self.out = self.call((self.input_layer1, self.input_layer2))
        self.build(input_shape=[(self.batch_size, self.n_state), (self.batch_size, self.n_latent_action)])
        self.summary()

        return Model(inputs=[self.input_layer1, self.input_layer2], outputs=self.out)

    def call(self, inputs, training=False):
        state = inputs[0]
        min_action = inputs[1]

        self.batch_size = state.shape[0]

        s_e = self.state_embedding(state)
        s_e = self.state_embedding_lrelu(s_e)

        a_e = self.action_embedding_dense(min_action)
        a_e = self.action_embedding_lrelu(a_e)

        concat = tf.concat([s_e, a_e], axis=-1)

        for i in range(len(self.action_remapping_dense)):
            concat = self.action_remapping_dense[i](concat)
            concat = self.action_remapping_lrelu[i](concat)
        
        remapped_action = self.action_remapping_dense_top(concat)
        remapped_action = self.action_remapping_softmax(remapped_action)

        return remapped_action
    
if __name__=="__main__":
    model_arm = ActionRemapNetwork(
        n_state=3,
        n_latent_action=3,
        n_action=5,
        units=128,
        units_p=64,
        layer_num=2,
        layer_num_p=1,
        batch_size=64,
        lrelu=0.2,
        )
    model_arm.build_graph()