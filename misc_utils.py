import numpy as np

def to_complex(two_channel_array, datashape='channels_first'):
    if datashape == 'channels_last':
        two_channel_array = two_channel_array.transpose()
    elif two_channel_array.shape[0] != 2:
        two_channel_array = two_channel_array.transpose()

    out = np.zeros((two_channel_array.shape[-1],1), dtype=np.complex)
    for i in range(two_channel_array.shape[-1]):
        out[i] = np.complex( two_channel_array[0][i], two_channel_array[1][i] )
    return out

