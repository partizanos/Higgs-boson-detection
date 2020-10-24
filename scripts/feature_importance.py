import numpy as np


def riemann_approximation_gaussian_cdf(b, N=100000):
    """Riemann approximation of gaussian distribution."""
    inte = 0
    a = -50
    for i in range(N):
        x = a+(b-a)*i/N
        inte += (np.exp(-x**2/2)*(b-a)/N)/np.sqrt(2*np.pi)
    return inte


def gaussian_test(y, feature):
    """Statistical test to quantify feature importance."""
    boson_index = np.where(y == 1)[0]

    boson_feature = feature[boson_index]
    negative_feature = feature[~boson_index]

    boson_mean = np.mean(boson_feature)
    negative_mean = np.mean(negative_feature)

    boson_var = (1/len(boson_feature))*np.var(boson_feature)
    negative_var = (1/len(negative_feature))*np.var(negative_feature)

    t_var = boson_var + negative_var
    t_mean = boson_mean - negative_mean
    test = t_mean/np.sqrt(t_var)

    # Under H0, test ~ N(0,1)
    return 2*(1-riemann_approximation_gaussian_cdf(abs(test)))


def skewness(x):
    """Compute the skewness of all features."""
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    skew = np.mean(((x - mean_x)/std_x)**3, axis=0)

    return skew

def significant_features(y, x, alpha=0.5):
    output = list()
    for i in range(x.shape[1]):
        output.append(gaussian_test(y,x[:,i]))
    output = (np.array(output)<alpha)*1
    
    return output