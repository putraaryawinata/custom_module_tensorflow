import tensorflow as tf

def GhostConv(input_layer, filter, kernel_size=1, stride=1, ratio=2, dw_size=3, act='relu'):
    init_channel = (filter // ratio) + 1
    new_channel = init_channel * (ratio-1)
    primary_conv = tf.keras.layers.Conv2D(init_channel, kernel_size=kernel_size, stride=stride, activation=act)(input_layer)
    cheap_operation = tf.keras.layers.Conv2D(new_channel, kernel_size=dw_size, stride=1, groups=init_channel, activation=act)(primary_conv)
    out = Concat([primary_conv, cheap_operation])
    return out[:,:,:,:filter]
