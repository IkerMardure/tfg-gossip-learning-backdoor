import sys
sys.path.insert(0, 'code')
try:
    import dataset
    print('dataset import OK')
except Exception as e:
    print('dataset import FAILED')
    import traceback
    traceback.print_exc()
