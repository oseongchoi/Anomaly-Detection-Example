import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import binned_statistic
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # Synthetic data generation.
    n_samples = 5000
    n_anomaly = 100
    mu = stats.norm(scale=3).rvs(4)
    mu = np.sort(mu)
    sigma = stats.uniform().rvs(4)

    # Latent features.
    z_normal = stats.norm(loc=mu[0], scale=sigma[0]).rvs(n_samples - n_anomaly)
    z_anomaly = stats.norm(loc=mu[3], scale=sigma[3]).rvs(n_anomaly)
    z = np.hstack([z_normal, z_anomaly])

    # Input features.
    x1 = stats.norm(loc=mu[1], scale=sigma[1]).rvs(n_samples)
    x2 = stats.uniform(loc=mu[2] - sigma[2], scale=mu[2] + sigma[2]).rvs(n_samples)

    # Basis expansions.
    features = np.stack([x1, x2, z], axis=1)
    features = np.hstack([
        features,
        # np.log1p(np.abs(features)),
        np.exp(features),
        np.power(features, -1),
        np.power(features, 2),
        np.power(features, 3),
        np.sin(features),
        np.cos(features),
        np.zeros((features.shape[0], 1))
    ])
    features = MinMaxScaler().fit_transform(features)

    # True weights.
    weights = stats.norm(scale=5).rvs(features.shape[1])

    # Visualize data distribution.
    # Independent variable.
    plt.figure(figsize=[16, 9])
    params = {'bins': 100, 'density': True, 'alpha': 0.5}
    plt.subplot(2, 5, 1, title='independent variables')
    plt.hist(z_normal, label='z_normal', **params)
    plt.hist(x1, label='x1', **params)
    plt.hist(x2, label='x2', **params)
    plt.hist(z_anomaly, label='z_anomaly', **params)
    plt.legend()

    # Dependent variable.
    plt.subplot(2, 5, 2, title='dependent variables')
    y_true = np.dot(features, weights)
    plt.hist(y_true[:-n_anomaly], label='y_normal', **params)
    plt.hist(y_true[-n_anomaly:], label='y_anomaly', **params)
    plt.legend()

    plt.subplot(2, 5, 3, title='x1 vs y_true')
    plt.plot(x1[:-n_anomaly], y_true[:-n_anomaly], '.', alpha=0.5)
    plt.plot(x1[-n_anomaly:], y_true[-n_anomaly:], '.', alpha=0.5, label='anomaly')
    plt.subplot(2, 5, 4, title='x2 vs y_true')
    plt.plot(x2[:-n_anomaly], y_true[:-n_anomaly], '.', alpha=0.5)
    plt.plot(x2[-n_anomaly:], y_true[-n_anomaly:], '.', alpha=0.5, label='anomaly')
    plt.subplot(2, 5, 5, title='z vs y_true')
    plt.plot(z[:-n_anomaly], y_true[:-n_anomaly], '.', alpha=0.5)
    plt.plot(z[-n_anomaly:], y_true[-n_anomaly:], '.', alpha=0.5, label='anomaly')
    plt.legend()

    # train a model.
    model = GradientBoostingRegressor()
    x = np.stack([x1, x2], axis=1)
    model.fit(x, y_true)

    # estimate the expectations of dependent variable.
    y_pred = model.predict(x)

    residual = y_true - y_pred

    # Visualize the result.
    plt.subplot(2, 5, 6, title='true vs pred')
    plt.plot(y_true[:-n_anomaly], y_pred[:-n_anomaly], '.', alpha=0.5)
    plt.plot(y_true[-n_anomaly:], y_pred[-n_anomaly:], '.', alpha=0.5, label='anomaly')
    plt.plot(y_true, y_true)
    plt.legend()

    plt.twinx()
    statistics, bin_edges, _ = binned_statistic(y_true, residual, 'std')
    plt.plot([np.mean(val) for val in zip(bin_edges[:-1], bin_edges[1:])], statistics, label='binned_std', color='red')
    plt.legend()

    plt.subplot(2, 5, 7, title='residuals')
    plt.hist(residual, bins=100)

    plt.subplot(2, 5, 8, title='x1 vs residual')
    plt.plot(x1[:-n_anomaly], residual[:-n_anomaly], '.', alpha=0.5)
    plt.plot(x1[-n_anomaly:], residual[-n_anomaly:], '.', alpha=0.5, label='anomaly')
    statistics, bin_edges, _ = binned_statistic(x1, residual, 'mean')
    plt.plot([np.mean(val) for val in zip(bin_edges[:-1], bin_edges[1:])], statistics, label='binned_mean')
    plt.legend()

    plt.twinx()
    statistics, bin_edges, _ = binned_statistic(x1, residual, 'std')
    plt.plot([np.mean(val) for val in zip(bin_edges[:-1], bin_edges[1:])], statistics, label='binned_std', color='red')
    plt.legend()

    plt.subplot(2, 5, 9, title='x2 vs residual')
    plt.plot(x2[:-n_anomaly], residual[:-n_anomaly], '.', alpha=0.5)
    plt.plot(x2[-n_anomaly:], residual[-n_anomaly:], '.', alpha=0.5, label='anomaly')
    statistics, bin_edges, _ = binned_statistic(x2, residual, 'mean')
    plt.plot([np.mean(val) for val in zip(bin_edges[:-1], bin_edges[1:])], statistics, label='binned_mean')
    plt.legend()

    plt.twinx()
    statistics, bin_edges, _ = binned_statistic(x2, residual, 'std')
    plt.plot([np.mean(val) for val in zip(bin_edges[:-1], bin_edges[1:])], statistics, label='binned_std', color='red')
    plt.legend()
    
    plt.subplot(2, 5, 10, title='z vs residual')
    plt.plot(z[:-n_anomaly], residual[:-n_anomaly], '.', alpha=0.5)
    plt.plot(z[-n_anomaly:], residual[-n_anomaly:], '.', alpha=0.5, label='anomaly')
    statistics, bin_edges, _ = binned_statistic(z, residual, 'mean')
    plt.plot([np.mean(val) for val in zip(bin_edges[:-1], bin_edges[1:])], statistics, label='binned_mean')
    plt.legend()

    plt.twinx()
    statistics, bin_edges, _ = binned_statistic(z, residual, 'std')
    plt.plot([np.mean(val) for val in zip(bin_edges[:-1], bin_edges[1:])], statistics, label='binned_std', color='red')
    plt.legend()

    plt.tight_layout()
    plt.show()
