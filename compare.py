import pickle
import numpy as np

cr = pickle.load(open('/tmp/cr_v4', 'rb'))
fr = pickle.load(open('/tmp/fr_v4', 'rb'))

length = len(cr['states'])

for i in range(length):
    print(f'{i=}')
    print(f'{cr["states"][i].flatten()[:20]=}')
    print(f'{fr["states"][i].flatten()[:20]=}')
    print(f'{np.all(cr["states"][i].flatten()[:20] == fr["states"][i].flatten()[:20])=}')
    print(f'{np.all(cr["states"][i] == fr["states"][i])=}')

import pdb; pdb.set_trace()
print('xx')
