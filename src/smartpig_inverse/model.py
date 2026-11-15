"""Untrained SmartPIG visual-transformation CNN architecture."""

import tensorflow as tf
from tensorflow.keras import Model, layers


class VisualTransformationLayer(layers.Layer):
    """
    Visual Transformation Layer following Lu et al. (2020).
    Creates multiple views: original, transposed, gradient, high-freq
    """

    def __init__(self, n_views=4, **kwargs):
        super().__init__(**kwargs)
        self.n_views = n_views

    def build(self, input_shape):
        self.view_weights = []
        for i in range(self.n_views):
            w = self.add_weight(
                name=f'view_{i}_weight',
                shape=(1,),
                initializer='ones',
                trainable=True
            )
            self.view_weights.append(w)

        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        self.sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        self.sobel_y = tf.transpose(self.sobel_x, [1, 0, 2, 3])

        laplacian = tf.constant([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=tf.float32)
        self.laplacian = tf.reshape(laplacian, [3, 3, 1, 1])

        super().build(input_shape)

    def call(self, inputs):
        views = []

        v1 = inputs * self.view_weights[0]
        views.append(v1)

        v2 = tf.transpose(inputs, [0, 2, 1, 3]) * self.view_weights[1]
        v2 = tf.image.resize(v2, [inputs.shape[1], inputs.shape[2]])
        views.append(v2)

        gx = tf.nn.conv2d(inputs, self.sobel_x, strides=[1,1,1,1], padding='SAME')
        gy = tf.nn.conv2d(inputs, self.sobel_y, strides=[1,1,1,1], padding='SAME')
        v3 = tf.sqrt(gx**2 + gy**2 + 1e-8) * self.view_weights[2]
        views.append(v3)

        v4 = tf.nn.conv2d(inputs, self.laplacian, strides=[1,1,1,1], padding='SAME')
        v4 = tf.abs(v4) * self.view_weights[3]
        views.append(v4)

        return tf.concat(views, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'n_views': self.n_views})
        return config


class MultiViewFusion(layers.Layer):
    """
    Multi-view feature fusion with attention-weighted combination.
    """

    def __init__(self, hidden_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.attention_dense = layers.Dense(self.hidden_dim, activation='tanh')
        self.attention_score = layers.Dense(1, activation='softmax')
        self.fusion_dense1 = layers.Dense(self.hidden_dim, activation='relu')
        self.fusion_bn = layers.BatchNormalization()
        self.fusion_dense2 = layers.Dense(self.hidden_dim, activation='relu')
        super().build(input_shape)

    def call(self, view_features):
        stacked = tf.stack(view_features, axis=1)
        att_hidden = self.attention_dense(stacked)
        att_scores = tf.nn.softmax(self.attention_score(att_hidden), axis=1)
        weighted = stacked * att_scores
        fused = tf.reduce_sum(weighted, axis=1)
        fused = self.fusion_dense1(fused)
        fused = self.fusion_bn(fused)
        fused = self.fusion_dense2(fused)
        return fused

    def get_config(self):
        config = super().get_config()
        config.update({'hidden_dim': self.hidden_dim})
        return config


def build_vt_cnn(
    input_shape=(5, 61, 1),
    output_shape=(61, 65),
    n_views=4,
    base_filters=32,
    feature_dim=256
):
    """Build VT-CNN (Visual Transformation CNN) model."""
    inputs = layers.Input(shape=input_shape, name='input')

    # Preprocessing
    x = layers.Resizing(16, 64, name='input_resize')(inputs)

    # Visual Transformation
    vt = VisualTransformationLayer(n_views=n_views, name='visual_transform')(x)

    # Multi-View Feature Extraction
    view_features = []

    for i in range(n_views):
        view = layers.Lambda(
            lambda x, idx=i: x[..., idx:idx+1],
            name=f'view_{i}_extract'
        )(vt)

        view = layers.Concatenate(name=f'view_{i}_expand')([view, view, view])

        # Block 1
        view = layers.Conv2D(base_filters, 3, padding='same', activation='relu',
                             name=f'view_{i}_b1_conv1')(view)
        view = layers.Conv2D(base_filters, 3, padding='same', activation='relu',
                             name=f'view_{i}_b1_conv2')(view)
        view = layers.BatchNormalization(name=f'view_{i}_b1_bn')(view)
        view = layers.MaxPooling2D(2, name=f'view_{i}_b1_pool')(view)

        # Block 2
        view = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu',
                             name=f'view_{i}_b2_conv1')(view)
        view = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu',
                             name=f'view_{i}_b2_conv2')(view)
        view = layers.BatchNormalization(name=f'view_{i}_b2_bn')(view)
        view = layers.MaxPooling2D(2, name=f'view_{i}_b2_pool')(view)

        # Block 3
        view = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu',
                             name=f'view_{i}_b3_conv1')(view)
        view = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu',
                             name=f'view_{i}_b3_conv2')(view)
        view = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu',
                             name=f'view_{i}_b3_conv3')(view)
        view = layers.BatchNormalization(name=f'view_{i}_b3_bn')(view)
        view = layers.MaxPooling2D(2, name=f'view_{i}_b3_pool')(view)

        # Global features
        view = layers.GlobalAveragePooling2D(name=f'view_{i}_gap')(view)
        view = layers.Dense(feature_dim, activation='relu', name=f'view_{i}_fc')(view)

        view_features.append(view)

    # Multi-View Fusion
    fused = MultiViewFusion(hidden_dim=feature_dim, name='fusion')(view_features)

    # Decoder
    x = layers.Dense(4 * 8 * base_filters * 4, activation='relu', name='dec_fc')(fused)
    x = layers.Reshape((4, 8, base_filters * 4), name='dec_reshape')(x)

    x = layers.Conv2DTranspose(base_filters * 4, 4, strides=2, padding='same',
                               activation='relu', name='dec_up1')(x)
    x = layers.BatchNormalization(name='dec_bn1')(x)

    x = layers.Conv2DTranspose(base_filters * 2, 4, strides=2, padding='same',
                               activation='relu', name='dec_up2')(x)
    x = layers.BatchNormalization(name='dec_bn2')(x)

    x = layers.Conv2DTranspose(base_filters, 4, strides=2, padding='same',
                               activation='relu', name='dec_up3')(x)
    x = layers.BatchNormalization(name='dec_bn3')(x)

    x = layers.Conv2DTranspose(base_filters // 2, 4, strides=2, padding='same',
                               activation='relu', name='dec_up4')(x)
    x = layers.BatchNormalization(name='dec_bn4')(x)

    x = layers.Resizing(output_shape[0], output_shape[1], name='output_resize')(x)

    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='output_conv1')(x)
    x = layers.Conv2D(16, 3, padding='same', activation='relu', name='output_conv2')(x)
    x = layers.Conv2D(1, 1, padding='same', activation='relu', name='output_depth')(x)

    outputs = layers.Reshape(output_shape, name='output')(x)

    model = Model(inputs, outputs, name='VT_CNN')
    return model
