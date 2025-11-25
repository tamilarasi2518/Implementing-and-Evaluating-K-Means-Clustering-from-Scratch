import numpy as np

def generate_data(n=100):
    np.random.seed(0)
    c1=np.random.randn(n,2)+[0,0]
    c2=np.random.randn(n,2)+[5,5]
    c3=np.random.randn(n,2)+[0,5]
    return np.vstack([c1,c2,c3])