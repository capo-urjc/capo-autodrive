import numpy as np

class RecordsTransform(object):
    """RecordsTransform .

    Args:
        control_keys: scalars to use as control
        state_keys: scalars to use from acceleration and velocity as states
    """
    def __init__(self,
                 control_keys=[], # No tenemos control ['brake', 'reverse', 'steer', 'throttle'],
                 state_keys=[ 'waypoints', 'acceleration', 'velocity']):

        self.control_keys = control_keys
        self.state_keys = state_keys

    def __call__(self, sample):
        records = sample['records']

        kps = records[['t.x', 't.y', 't.z']]
        kps = np.asarray((kps - kps.iloc[0]).to_numpy(), dtype=np.float32) # relative positions
        sample['kps'] = kps

        # sample['state'] = [] # añadir velocidad y aceleración
        del (sample)['records'] # Esto hay que quitarlo porque no se puede iterar dataframes


    # ctl = []
        # stt = []
        # for seq in records:
        #     ctl.append([seq['control'][k] for k in self.control_keys])
        #     stt.append([list(seq['state'][k].values()) for k in self.state_keys])

        # ctl = np.asarray(ctl) # [# of seq x 4] --> 'brake', 'reverse', 'steer', 'throttle'
        # stt = np.concatenate([np.asarray(stt)[:,i,:] for i in range(len(stt[0]))], axis=1) # of seq x 4] --> acceleration=[value, x,y,z], velocity=[value, x,y,z]
        # sample['control'] = np.concatenate([ctl, stt], axis=1) # [seq, 12 scalars = 4 control + 8 state])
        return sample

class ImageNormalization(object):
    """ImageNormalization .

    Args:
        value
    """
    def __init__(self, value=255.0, stats=None):

        self.value = value
        self.stats = {'mean': [0.5435199558323847, 0.5386219601332897, 0.5325046995624928],
                      'std_dev': [0.1990361211957141, 0.20639664185617518, 0.22376878168172593]} if stats is None else stats

    def __call__(self, sample):

        for key in sample:
            if 'rgb' in key:
                if sample[key].ndim == 4:
                    sample[key] = (sample[key]/self.value).astype(np.float32)
                    sample[key] = (sample[key] - np.asarray(self.stats['mean'], dtype=np.float32)[:,None,None])/ np.asarray(self.stats['std_dev'], dtype=np.float32)[:, None, None]
        return sample