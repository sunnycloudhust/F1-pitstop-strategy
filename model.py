import tensorflow as tf
from keras.layers import Bidirectional, LSTM, Dense, Dropout


class F1PitStopPredictor(tf.keras.Model):
    """
    2-layer Bidirectional LSTM classifier for binary pit-stop prediction.

    Input : (batch, seq_length, n_features)
    Output: (batch, 1)  — sigmoid probability of a pit stop on the last lap
    """

    def __init__(self):
        super().__init__()

        self.bilstm_1  = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2))
        self.dropout_1 = Dropout(0.3)

        self.bilstm_2  = Bidirectional(LSTM(32, return_sequences=False, recurrent_dropout=0.2))
        self.dropout_2 = Dropout(0.3)

        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.bilstm_1(inputs, training=training)
        x = self.dropout_1(x,      training=training)
        x = self.bilstm_2(x,       training=training)
        x = self.dropout_2(x,      training=training)
        return self.classifier(x)