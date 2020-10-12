"""
plot_effectiveness.py
---------------------

Figure showing the overall effectiveness of the algorithm.

"""

import pandas as pd
import matplotlib.pyplot as plt


def read_fig1():
    """Read from plot-data."""
    df1 = pd.read_csv('plot-data/Email/Email_equalAlpha.csv')
    df2 = pd.read_csv('plot-data/Airport/Airport_equalAlpha.csv')
    df3 = pd.read_csv('plot-data/Brain/Brain_equalAlpha.csv')
    df1['graph'] = 'Email'
    df2['graph'] = 'Airport'
    df3['graph'] = 'Brain'
    return pd.concat([df1, df2, df3])


def main():
    """Make a pretty plot."""
    df = read_fig1()

    _, axes = plt.subplots(1, 3, figsize=(4.5, 1.6), sharey=True)
    for idx, graph in enumerate(['Email', 'Airport', 'Brain']):
        data = df[df.graph == graph]
        axes[idx].errorbar(data.budget,
                           data['targeted infectious ratio'],
                           yerr=data['targeted std-error'])
        axes[idx].errorbar(data.budget,
                           data['nontargeted infectious ratio'],
                           yerr=data['nontargeted std-err'])
        axes[idx].set_xlabel(r'$\gamma$')
    axes[0].set_ylabel('$I_{modified} - I_{original}$')
    plt.show()


if __name__ == '__main__':
    main()
