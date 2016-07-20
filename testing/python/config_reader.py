from configobj import ConfigObj
import numpy as np


def config_reader():
    config = ConfigObj('config')

    param = config['param']
    model_id = param['modelID']
    model = config['models'][model_id]
    model['boxsize'] = int(model['boxsize'])
    model['np'] = int(model['np'])
    num_limb = len(model['limbs'])/2
    model['limbs'] = np.array(model['limbs']).reshape((num_limb, 2))
    model['limbs'] = model['limbs'].astype(np.int)
    model['sigma'] = float(model['sigma'])
    #param['starting_range'] = float(param['starting_range'])
    #param['ending_range'] = float(param['ending_range'])
    #param['octave'] = int(param['octave'])
    param['use_gpu'] = int(param['use_gpu'])
    param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])

    return param, model

if __name__ == "__main__":
    config_reader()
