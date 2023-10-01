import tensorflow as tf
import tensorflow_hub as hub


class ToxicCommentClassifier(tf.keras.Model):
    def __init__(self):
        super(ToxicCommentClassifier, self).__init__()
        self.bert_preprocess = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
        self.bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        text_input = inputs
        preprocessed_text = self.bert_preprocess(text_input)
        bert_output = self.bert_encoder(preprocessed_text)
        dropout = self.dropout(bert_output["pooled_output"])
        dense_output = self.dense1(dropout)
        dense_output = self.dense2(dense_output)

        return dense_output
