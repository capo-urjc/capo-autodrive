import numpy as np

class ControlTransform(object):
    """ControlTransform .

    Args:
        control_keys: scalars to use as control
        state_keys: scalars to use from acceleration and velocity as states
    """
    def __init__(self,
                 control_keys=['brake', 'reverse', 'steer', 'throttle'],
                 state_keys=['acceleration', 'velocity']):

        self.control_keys = control_keys
        self.state_keys = state_keys

    def __call__(self, sample):
        control = sample['control']

        ctl = []
        stt = []
        for seq in control:
            ctl.append([seq['control'][k] for k in self.control_keys])
            stt.append([list(seq['state'][k].values()) for k in self.state_keys])


        ctl = np.asarray(ctl) # [# of seq x 4] --> 'brake', 'reverse', 'steer', 'throttle'
        stt = np.concatenate([np.asarray(stt)[:,i,:] for i in range(len(stt[0]))], axis=1) # of seq x 4] --> acceleration=[value, x,y,z], velocity=[value, x,y,z]
        sample['control'] = np.concatenate([ctl, stt], axis=1) # [seq, 12 scalars = 4 control + 8 state])
        return sample

class ImageNormalization(object):
    """ImageNormalization .

    Args:
        value
    """
    def __init__(self, value=255.0):

        self.value = value

    def __call__(self, sample):
        for key in sample:
            if 'rgb' in key:
                sample[key] = (sample[key]/self.value).astype(np.float32)
        return sample