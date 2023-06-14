import tensorflow as tf

def channel_attention_module(input_layer, filter, ratio=8):
    pool = input_layer.get_shape()[1:-1]
    avgpool = tf.keras.layers.AvgPool2D(pool)(input_layer)
    maxpool = tf.keras.layers.MaxPool2D(pool)(input_layer)

    mlp1 = tf.keras.layers.Dense(filter//ratio)(avgpool)
    mlp1 = tf.keras.layers.Dense(filter)(mlp1)

    mlp2 = tf.keras.layers.Dense(filter//ratio)(maxpool)
    mlp2 = tf.keras.layers.Dense(filter)(mlp2)

    mlp = tf.keras.layers.Add(activation='relu')([mlp1, mlp2])
    mlp = tf.nn.relu(mlp)
    return tf.keras.layers.Multiply()([input_layer, mlp])

def spatial_attention_module(input_layer, kernel_size=7):
    pool = input_layer.get_shape()[1:-1]
    avgpool = tf.keras.layers.AvgPool2D(pool)(input_layer)
    maxpool = tf.keras.layers.MaxPool2D(pool)(input_layer)
    x = Concat([avgpool, maxpool])
    x = tf.keras.layers.Conv2D(1, padding='same',
                               use_bias=False,
                               kernel_size=kernel_size,
                               activation='sigmoid'
                               )(x)
    return tf.keras.layers.Multiply()([input_layer, x])

def cbam(input_layer, filter, ratio=8, kernel_size=7):
    cam = channel_attention_module(input_layer, filter, ratio=ratio)
    sam = spatial_attention_module(cam, kernel_size=kernel_size)
    return sam
