import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.platform import tf_logging as logging
import os
import json
"""
I train an LSTM model on the old (single pin, 30 tests) dataset. Classifying
between damaged and undamaged. Using the foiled-16 feature method as in the
state estimation project.
"""
#%%
class TrainingGenerator(keras.utils.Sequence):
    
    def __init__(self, *args, train_len=400):
        self.args = args
        self.train_len = train_len
        self.length = args[0].shape[1]//train_len
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # F_batch = self.F[:,index*self.train_len:(index+1)*self.train_len,:]
        # lstm_batch = self.lstm_input[:,index*self.train_len:(index+1)*self.train_len,:]
        # return lstm_batch, F_batch
        rtrn = [arg[:,index*self.train_len:(index+1)*self.train_len,:] for arg in self.args]
        
        return rtrn[:-1], rtrn[-1] 
"""
Reset all stateful layers in a model
"""
class StateResetter(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        for layer in self.model.layers:
            if(layer.stateful):
                layer.reset_states()
    
"""
Basically EarlyStopping's restore_best_weights = True. Restore best
weights at the end of training.
"""
class RestoreBestWeights(keras.callbacks.Callback):
    
    def __init__(self, monitor="val_loss", mode="auto"):
        super().__init__()
        
        self.monitor = monitor
        self.best_weights = None
        
        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        
    
    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value, reference_value)
    
    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Restore best weights conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value
    
    def on_train_begin(self, logs=None):
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        
        
        if(self.best_weights is None):
            self.best_weights = self.model.get_weights()
        
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
#%% load data
X = np.load('./datasets/X.npy')
y = np.load('./datasets/y.npy')
train_len = X.shape[1]

training_generator = TrainingGenerator(X, y, train_len=train_len)
#%%
plt.figure()
plt.plot(X[:,:,0].T)
#%% train
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, stateful=True, batch_input_shape=[X.shape[0], None, 1]),
    keras.layers.LSTM(50, return_sequences=True, stateful=True),
    keras.layers.TimeDistributed(keras.layers.Dense(2, activation='softmax', use_bias=False))
])
adam = keras.optimizers.Adam(
    learning_rate=0.000001,
)
model.compile(
    loss='mse',
    optimizer=adam,
)

state_resetter = StateResetter()
restore_best_weights = RestoreBestWeights()
#%%
# model = keras.models.load_model('./model saves/health pred model')
#%% train model
for layer in model.layers:
    if(layer.stateful):
        layer.reset_states()
history = model.fit(
    training_generator,
    shuffle=False,
    epochs=1000,
    validation_data=[X, y],
    callbacks=[state_resetter, restore_best_weights]
)
history = history.history
#%%
# plot validation loss and validation loss through training
plt.figure()
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val loss')
plt.legend()
plt.tight_layout()

for layer in model.layers:
    if(layer.stateful):
        layer.reset_states()
y_pred = model.predict(X)
mse = np.mean((y_pred - y)**2)
print('mse: ' + str(mse))

for i, y_p in enumerate(y_pred):
    expected = 'healthy'
    if(i > 10):
        expected = 'unhealthy'
    plt.figure()
    plt.plot(y_p[:,0], label='probability healthy')
    plt.plot(y_p[:,1], label='probability unhealthy')
    plt.ylim((0, 1))
    plt.legend(loc=1)
    plt.title('expected: ' + expected)
#%% save model
# model.save('./model saves/health pred model')