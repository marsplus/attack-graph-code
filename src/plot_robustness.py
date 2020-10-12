"""
plot_robustness.py
------------------

Figure showing certified robustness results.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

COLORS = {'BA': 'C0', 'BTER': 'C1', 'WS': 'C2'}


def read_fig2():
    """Read from plot-data."""
    df1 = pd.read_csv('plot-data/Figure2/BA_robustness.csv', index_col=0)
    df2 = pd.read_csv('plot-data/Figure2/BTER_robustness.csv', index_col=0)
    df3 = pd.read_csv('plot-data/Figure2/Small-World_robustness.csv', index_col=0)
    df1['graph'] = 'BA'
    df2['graph'] = 'BTER'
    df3['graph'] = 'WS'

    vlines = pd.read_csv('plot-data/Figure2/robustness_bound.txt', index_col=0)

    return pd.concat([df1, df2, df3]), vlines


def main():
    """Make a pretty plot."""
    df, vlines = read_fig2()

    plt.figure(figsize=(2.5, 1.6))
    ax = plt.gca()
    for graph in ['BA', 'BTER', 'WS']:
        data = df[df.graph == graph]
        ax.plot(data.index, data['infectious ratio'], 'o-', label=graph,
                color=COLORS[graph], ms=3, lw=1)
        ax.axvline(vlines.loc[graph, 'bound'], color=COLORS[graph], ls='--')
    ax.legend(bbox_to_anchor=(1.05, 0.95), loc='upper left')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('$I_{modified} - I_{original}$')
    ax.grid(linestyle='--')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    plt.savefig('pics/fig_robustness.png', dpi=300, bbox_inches='tight')
    plt.savefig('pics/fig_robustness.pdf', dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    main()
