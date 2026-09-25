"""Plot an existing study report; no extra simulation or acceptance inference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(source, output):
    report = json.loads(Path(source).read_text(encoding='utf-8'))
    cases = report['cases']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name in ('baseline', 'time_half', 'time_quarter'):
        r = cases.get(name)
        if not r or not r['history']:
            continue
        t = [0., *[row['time_s']*1000 for row in r['history']]]
        v0 = r['initial']['cavity_volume_ml']
        delta = [0., *[row['cavity_volume_ml']-v0 for row in r['history']]]
        axes[0, 0].plot(t, delta, label=f"dt={r['case']['dt']:g} s")
    axes[0, 0].set(title='Time-step refinement: same physical duration', xlabel='Time [ms]', ylabel='Cavity volume change [mL]')
    axes[0, 0].legend(fontsize=9)
    names = [n for n in ('solid_coarse', 'baseline', 'solid_fine', 'fluid_8', 'fluid_10')
             if n in cases and cases[n]['completed']]
    if names:
        axes[0, 1].bar(range(len(names)), [cases[n]['summary']['delta_cavity_ml'] for n in names], color='#4386a0')
        axes[0, 1].set_xticks(range(len(names)), names, rotation=20, ha='right')
    final_ms = cases['baseline']['case']['final_time']*1000
    axes[0, 1].set(title=f'Separate solid/fluid mesh changes at {final_ms:g} ms', ylabel='Final cavity volume change [mL]')
    for name in ('baseline', 'duration_005', 'duration_010'):
        r = cases.get(name)
        if not r or not r['history']:
            continue
        axes[1, 0].plot([0., *[row['time_s']*1000 for row in r['history']]],
            [1., *[row['minimum_detF'] for row in r['history']]], label=f"T={r['case']['final_time']:g}s"+(' FAILED' if not r['completed'] else ''))
    axes[1, 0].set(title='Duration extension: sampled minimum det(F)', xlabel='Time [ms]', ylabel='Minimum det(F)')
    axes[1, 0].legend(fontsize=9)
    assessment = report.get('assessment') or {}
    labels, volume_errors, displacement_errors = [], [], []
    for name, a in assessment.get('axes', {}).items():
        metrics = a['finest_pair_relative_metrics']
        dv, du = metrics.get('delta_cavity_relative_difference'), metrics.get('max_displacement_relative_difference')
        if dv is not None and du is not None:
            labels.append(name.replace('_', '\n'))
            volume_errors.append(100*dv)
            displacement_errors.append(100*du)
    axes[1, 1].bar([i-.18 for i in range(len(labels))], volume_errors, width=.35, color='#4386a0', label='Volume response')
    axes[1, 1].bar([i+.18 for i in range(len(labels))], displacement_errors, width=.35, color='#c17e47', label='Max displacement')
    axes[1, 1].set_xticks(range(len(labels)), labels, fontsize=9)
    axes[1, 1].axhline(100*assessment.get('relative_response_screen_tolerance', .05), color='firebrick', ls='--', label='5% diagnostic screen')
    axes[1, 1].set(title='Finest pair: relative response differences', ylabel='Difference [%]')
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Generated LV numerical study | short ramp, not a cardiac cycle', fontsize=15)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/lv_study/study.json')
    parser.add_argument('--output', default='results/lv_study/study.png')
    args = parser.parse_args()
    plot(args.source, args.output)
