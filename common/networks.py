from keras.models import Sequential
from keras.layers import Dense, Activation

class MLP:
    def __init__(self, sess, obs, act, scope='MLP'):
        self.sess = sess
        self.obs = obs
        self.act = act
        # TODO: Get obs and act dims
        self.obs_dim = 5
        self.act_dim = 5
        self.name = 'discrete_policy'
        scope = scope
        self.nn_output = []
        with tf.variable_scope(scope):
            model = Sequential()
            model.add(Dense(32, activation='tanh', input_dim=self.obs_dim))
            model.add(Dense(32, activation='tanh'))
            self.nn_output.append(model.add(Dense(self.act_dim, activation='tanh')))

        self.vars = tf.trainable_variables(scope=scope)