import tensorflow as tf
from keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization

class F1PitStopPredictor(tf.keras.Model):
    """
    Optimized 2-layer Bidirectional LSTM classifier for binary pit-stop prediction.
    """
    def __init__(self):
        super().__init__()

        # Lớp BiLSTM 1 - Tăng units lên 128
        self.bilstm_1 = Bidirectional(LSTM(128, return_sequences=True))
        self.bn_1 = BatchNormalization()
        self.dropout_1 = Dropout(0.3)

        # Lớp BiLSTM 2 - Tăng units lên 64
        self.bilstm_2 = Bidirectional(LSTM(64, return_sequences=False))
        self.bn_2 = BatchNormalization()
        self.dropout_2 = Dropout(0.3)

        # Classifier
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.bilstm_1(inputs, training=training)
        x = self.bn_1(x, training=training)
        x = self.dropout_1(x, training=training)
        
        x = self.bilstm_2(x, training=training)
        x = self.bn_2(x, training=training)
        x = self.dropout_2(x, training=training)
        
        return self.classifier(x)
