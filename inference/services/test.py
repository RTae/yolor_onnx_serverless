import ctypes
import os

exlibpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/lib/'

print(exlibpath)
# exlibpath = '/home/site/wwwroot/lib/'
ctypes.CDLL(exlibpath + 'libglib-2.0.so.0')
ctypes.CDLL(exlibpath + 'libgthread-2.0.so.0')