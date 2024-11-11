import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def zero_inflated_lognormal_pred(logits:tf.Tensor, clip_value_logn_sigma:float=1.0)->tf.Tensor:
    """
    Calculates predicted mean of zero inflated lognormal logits.

    Args:
        logits: [batch_size, 3] tensor of logits.
        clip_value_logn_sigma (float): Value at which the sigma parameter of the LogN distribution is clipped.

    Returns:
        preds: [batch_size, 1] tensor of predicted mean.
    """
    logits = tf.convert_to_tensor(logits, dtype=tf.float32)
    positive_probs = tf.keras.backend.sigmoid(logits[:, 0])
    loc = logits[:, 1]
    scale = tf.math.minimum(
        tf.keras.backend.softplus(logits[:, 2]), 
        clip_value_logn_sigma
    )
    preds = (positive_probs*tf.keras.backend.exp(loc + 0.5*tf.keras.backend.square(scale)))
    return positive_probs, preds



class ZeroInflatedLogNormalLoss(tf.keras.losses.Loss):
    def __init__(self, 
                 weigh_high_cltv_samples:bool=False, 
                 cltv_upp_threshold:float=10000.0, 
                 high_cltv_samples_weight:float=10.0, 
                 clip_value_logn_sigma:float=1.0, 
                 **kwargs):
        super().__init__(**kwargs)
        self.weigh_high_cltv_samples = weigh_high_cltv_samples
        self.cltv_upp_threshold = cltv_upp_threshold
        self.high_cltv_samples_weight = high_cltv_samples_weight
        self.clip_value_logn_sigma = clip_value_logn_sigma

    def call(self, labels:np.ndarray, logits:np.ndarray):
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        positive = tf.cast(labels>0, tf.float32)
        skewed_weights = tf.where(labels>self.cltv_upp_threshold, self.high_cltv_samples_weight, 1.0)

        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        assert logits.shape[-1] == 3

        # Logits
        positive_logits = logits[:, 0]
        classification_loss = tf.keras.losses.binary_crossentropy(
            y_true=positive, y_pred=positive_logits, from_logits=True
        )

        # LogNormal Location
        loc = logits[:, 1]

        # LogNormal Sigma (Softflus Activation + Clipping done)
        sigma = tf.math.minimum(
            tf.math.maximum(
                tf.keras.backend.softplus(logits[:, 2]), 
                tf.math.sqrt(tf.keras.backend.epsilon())
            ), 
            self.clip_value_logn_sigma
        )

        # Safe Labels
        safe_labels = positive * labels + (1 - positive) * tf.keras.backend.ones_like(labels)

        if self.weigh_high_cltv_samples:
            regression_loss = -tf.keras.backend.mean(
                positive * skewed_weights * tfd.LogNormal(loc=loc, scale=sigma).log_prob(safe_labels), axis=-1
            )
        else:
            regression_loss = -tf.keras.backend.mean(
                positive * tfd.LogNormal(loc=loc, scale=sigma).log_prob(safe_labels), axis=-1
            )

        return classification_loss + regression_loss







# def zero_inflated_lognormal_loss(labels:np.ndarray, 
#                                  logits:np.ndarray,
#                                  weigh_high_cltv_samples:bool=False,
#                                  cltv_upp_threshold:float=10000.0, 
#                                  high_cltv_samples_weight:float=10.0, 
#                                  clip_value_logn_sigma:float=1.0)->float:

#     labels = tf.convert_to_tensor(labels, dtype=tf.float32)
#     positive = tf.cast(labels > 0, tf.float32)
#     skewed_weights = tf.cast(tf.where(labels>cltv_upp_threshold, high_cltv_samples_weight, 1), tf.float32)
    
#     logits = tf.convert_to_tensor(logits, dtype=tf.float32)
#     assert logits.shape[-1] == 3

#     # Logits
#     positive_logits = logits[:, 0]
#     classification_loss = tf.keras.losses.binary_crossentropy(y_true=positive, y_pred=positive_logits, from_logits=True)
    
#     # LogNormal Location
#     loc = logits[:, 1]

#     # LogNormal Sigma (Softflus Activation + Clipping done)
#     sigma = tf.math.minimum(tf.math.maximum(tf.keras.backend.softplus(logits[:, 2]), tf.math.sqrt(tf.keras.backend.epsilon())), clip_value_logn_sigma)
    

#     safe_labels = positive*labels + (1 - positive)*tf.keras.backend.ones_like(labels)
#     if weigh_high_cltv_samples:
#         regression_loss = -tf.keras.backend.mean(positive * skewed_weights * tfd.LogNormal(loc=loc, scale=sigma).log_prob(safe_labels), axis=-1)
#     else:
#         regression_loss = -tf.keras.backend.mean(positive * tfd.LogNormal(loc=loc, scale=sigma).log_prob(safe_labels), axis=-1)
    
    
#     ziln_loss = classification_loss + regression_loss
#     return ziln_loss
    