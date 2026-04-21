import numpy as np
import pandas
def get_data(function, num_samples, noise, seed):
    X, y, gt = function(num_samples =num_samples, seed = seed)
    rng = np.random.default_rng(seed)
    y_std = np.std(y)
    y_noisy = y + rng.normal(0.0, noise * y_std, size=y.shape)
    return X, y_noisy, gt

def f1(num_samples = 30000, seed = 42):
    np.random.seed(seed)
    X = np.empty((num_samples, 10))
    X[:, [0, 1, 2, 5, 6, 8]] = np.random.uniform(0.0, 1.0, (num_samples, 6))
    X[:, [3, 4, 7, 9]] = np.random.uniform(0.6, 1.0, (num_samples, 4))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X.T

    y = (np.pi**(x1 * x2) * np.sqrt(2 * x3) - 
         np.arcsin(x4) + 
         np.log(x3 + x5) - 
         (x9 / x10) * np.sqrt(x7 / x8) - 
         x2 * x7)
    
    interactions = [{1, 2, 3}, {3, 5}, {7, 8, 9, 10}, {2, 7}]
    
    return X, y, interactions

def f2(num_samples = 30000, seed = 42):
    np.random.seed(seed)
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 10))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X.T
    
    y = (np.pi**(x1 * x2) * np.sqrt(2 * np.abs(x3)) - 
         np.arcsin(0.5 * x4) + 
         np.log(np.abs(x3 + x5) + 1) + 
         (x9 / (1 + np.abs(x10))) * np.sqrt(np.abs(x7) / (1 + np.abs(x8))) - 
         x2 * x7)
    
    interactions = [{1, 2, 3}, {3, 5}, {7, 8, 9, 10}, {2, 7}]
    return X, y, interactions

def f3(num_samples = 30000, seed = 42):
    
    np.random.seed(seed)
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 10))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X.T
    
    y = (np.exp(np.abs(x1 - x2)) + 
         np.abs(x2 * x3) - 
         (x3**2)**np.abs(x4) +  # Change base to x3**2 to handle negatives
         np.log(x4**2 + x5**2 + x7**2 + x8**2) + 
         x9 + 
         1 / (1 + x10**2))
    
    interactions = [{1, 2}, {2, 3}, {3, 4}, {4, 5, 7, 8}]
    return X, y, interactions

def f4(num_samples = 30000, seed = 42):
    np.random.seed(seed)
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 10))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X.T
    
    y = (np.exp(np.abs(x1 - x2)) + 
         np.abs(x2 * x3) - 
         (x3**2)**np.abs(x4) + 
         (x1 * x4)**2 + 
         np.log(x4**2 + x5**2 + x7**2 + x8**2) + 
         x9 + 
         1 / (1 + x10**2))
    
    interactions = [{1, 2}, {2, 3}, {3, 4}, {1, 4}, {4, 5, 7, 8}]
    return X, y, interactions

def f5(num_samples = 30000, seed = 42):
    np.random.seed(seed)
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 10))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X.T
    
    y = (1 / (1 + x1**2 + x2**2 + x3**2) + 
         np.sqrt(np.exp(x4 + x5)) + 
         np.abs(x6 + x7) + 
         x8 * x9 * x10)
    
    interactions = [{1, 2, 3}, {4, 5}, {6, 7}, {8, 9, 10}]
    return X, y, interactions

def f6(num_samples = 30000, seed = 42):
    np.random.seed(seed)
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 10))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X.T
    
    y = (np.exp(np.abs(x1 * x2) + 1) - 
         np.exp(np.abs(x3 + x4) + 1) + 
         np.cos(x5 + x6 - x8) + 
         np.sqrt(x8**2 + x9**2 + x10**2))
    
    interactions = [{1, 2}, {3, 4}, {5, 6, 8}, {8, 9, 10}]
    return X, y, interactions

def f7(num_samples = 30000, seed = 42):
    np.random.seed(seed)
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 10))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X.T
    
    y = ((np.arctan(x1) + np.arctan(x2))**2 + 
         np.maximum(x3 * x4 + x6, 0) - 
         1 / (1 + (x4 * x5 * x6 * x7 * x8)**2) + 
         (np.abs(x7) / (1 + np.abs(x9)))**5 + 
         np.sum(X, axis=1))
    
    interactions = [{1, 2}, {3, 4, 6}, {4, 5, 6, 7, 8}, {7, 9}]
    return X, y, interactions

def f8(num_samples = 30000, seed = 42):
    np.random.seed(seed)
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 10))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X.T
    
    y = (x1 * x2 + 
         2**(x3 + x5 + x6) + 
         2**(x3 + x4 + x5 + x7) + 
         np.sin(x7 * np.sin(x8 + x9)) + 
         np.arccos(0.9 * x10))
    
    interactions = [{1, 2}, {3, 5, 6}, {3, 4, 5, 7}, {7, 8, 9}]
    return X, y, interactions

def f9(num_samples = 30000, seed = 42):
    np.random.seed(seed)
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 10))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X.T
    
    y = (np.tanh(x1 * x2 + x3 * x4) * np.sqrt(np.abs(x5)) + 
         np.exp(x5 + x6) + 
         np.log((x6 * x7 * x8)**2 + 1) + 
         x9 * x10 + 
         1 / (1 + np.abs(x10)))
    
    interactions = [{1, 2, 3, 4, 5}, {5, 6}, {6, 7, 8}, {9, 10}]
    return X, y, interactions

def f10(num_samples = 30000, seed = 42):
    np.random.seed(seed)
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 10))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X.T
    
    y = (np.sinh(x1 + x2) + 
         np.arccos(np.tanh(x3 + x5 + x7)) + 
         np.cos(x4 + x5) + 
         1 / np.cos(x7 * x9))
    
    interactions = [{1, 2}, {3, 5, 7}, {4, 5}, {7, 9}]
    return X, y, interactions

functions = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]

def main():
    #just a test
    data = get_data(f1, num_samples = 3000, noise = 0.1, seed = 42)
    print("X: ", data[0][:5])
    print("Y: ", data[1][:5])
    print("GT: ", data[2])
if __name__ == "__main__":
    main()



