"""Plot the frozen diagnostic report, without running or modifying a solver."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(source, output):
    report = json.loads(Path(source).read_text(encoding='utf-8'))
    if not report['completed']:
        raise ValueError('complete diagnosis required for comparison plot')
    fig, axes = plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    labels = ['native / density','fixed / density','native / dual','fixed / dual']
    keys = ['native_density','fixed_density','native_dual','fixed_dual']
    x = np.arange(4)
    for shift,projection,color in [(-.25,'tentative','#7a8793'),(0,'chorin','#cf6e32'),(.25,'schur','#2b7a78')]:
        values = [report['comparisons'][k+'_'+projection][-1]['solid_velocity_relative_difference']*100 for k in keys]
        bars = axes[0].bar(x+shift,values,.23,label=projection,color=color)
        axes[0].bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
    a,b = report['metadata']['levels'][-2:]
    axes[0].set_title(f'Solid velocity change: {a}³ to {b}³')
    axes[0].set_ylabel('Relative nodal norm difference (%)')
    axes[0].set_xticks(x,labels,rotation=20,ha='right')
    axes[0].legend(fontsize=8)
    for k,label in zip(keys,labels):
        levels = report['metadata']['levels']
        cases = [report['cases'][f'{k}_{n}'] for n in levels]
        axes[1].plot(levels,[c['responses']['chorin']['cavity_rate_ml_per_s'] for c in cases],'-o',label=label)
        axes[2].semilogy(levels,[c['responses']['chorin']['weak_divergence_relative_to_tentative'] for c in cases],'-o',label=label)
    axes[1].set_title('Frozen response after Chorin')
    axes[1].set_ylabel('Cavity volume rate (mL/s)')
    axes[2].set_title('Remaining weak divergence after Chorin')
    axes[2].set_ylabel('||D u|| / ||D u*|| (same mesh)')
    for ax in axes[1:]:
        ax.set_xlabel('Fluid cells per axis')
        ax.set_xticks(report['metadata']['levels'])
        ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Frozen geometry and force; diagnostic alternatives, not a converged cardiac cycle',fontsize=12)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='results/ib_diagnosis/diagnosis.json')
    parser.add_argument('--output',default='results/ib_diagnosis/diagnosis.png')
    args = parser.parse_args()
    plot(args.source,args.output)
