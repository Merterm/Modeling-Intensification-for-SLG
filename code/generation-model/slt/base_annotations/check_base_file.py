import gzip
import pickle 

import shutil
with gzip.open('../new_data/phoenix14t.yang.dev', 'rb') as f_in:
    loaded_object = pickle.load(f_in)
    print(type(loaded_object))
    print(loaded_object[0])
    sign_shape = loaded_object[0]['sign'].shape
    sign_shape = loaded_object[1]['sign'].shape
    print(sign_shape)