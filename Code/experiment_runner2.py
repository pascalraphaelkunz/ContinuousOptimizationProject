from joblib import Parallel, delayed
import newton
from time import time

def run_native_newton(*args, **kwargs):
    newton.run(*args, **kwargs)

if __name__ == '__main__':
    start = time()
    # Define the arguments for each function call
    ijcnn_args = (10, 'Datasets/ijcnn/train', 'Datasets/ijcnn/test')
    a9a_args = (10, 'Datasets/a9a/train.txt', 'Datasets/a9a/test.txt')
        

    # Create a list of argument tuples
    args_list = [(ijcnn_args, {'type': 'ijcnn'}), (a9a_args, {'type': 'a9a'})]

    # Execute the function calls in parallel
    Parallel(n_jobs=2)(delayed(run_native_newton)(*args, **kwargs) for args, kwargs in args_list)
    print(time()-start)