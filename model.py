import tensorflow as tf
import keras

from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam

class F1PitStopPredictor(tf.keras.Model):
    def __init__(self):
        super(F1PitStopPredictor, self).__init__()
        
        # Layer 1: 256 units, returns sequences for the next layer
        self.bilstm_1 = Bidirectional(
            LSTM(256, return_sequences=True, recurrent_dropout=0.2)
        )
        self.dropout_1 = Dropout(0.2)
        
        # Layer 2: 128 units, returns sequences for the next layer
        self.bilstm_2 = Bidirectional(
            LSTM(128, return_sequences=True, recurrent_dropout=0.2)
        )
        self.dropout_2 = Dropout(0.3)
        
        # Layer 3: 64 units, returns only the final output for the dense layer
        self.bilstm_3 = Bidirectional(
            LSTM(64, return_sequences=False, recurrent_dropout=0.2)
        )
        self.dropout_3 = Dropout(0.3)
        
        # Output Layer: Binary classification (Pit Stop or No Pit Stop)
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # The forward pass
        # The 'training' boolean ensures Dropout is only active during training, not inference
        x = self.bilstm_1(inputs, training=training)
        x = self.dropout_1(x, training=training)
        
        x = self.bilstm_2(x, training=training)
        x = self.dropout_2(x, training=training)
        
        x = self.bilstm_3(x, training=training)
        x = self.dropout_3(x, training=training)
        
        return self.classifier(x)

# --- How to instantiate and compile the model ---

# Create an instance of the class
pit_stop_model = F1PitStopPredictor()

# Define the optimizer as specified in the paper
custom_optimizer = Adam(learning_rate=5e-4)

# Compile the model
pit_stop_model.compile(
    optimizer=custom_optimizer, 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# Optional: Build the model explicitly to see the summary before training
# Assuming 10 timesteps and X features (replace X with your actual feature count)
# pit_stop_model.build(input_shape=(None, 10, X))
# pit_stop_model.summary()