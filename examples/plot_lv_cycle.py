"""Plot prescribed loading and computed volume from a saved LV cycle run."""
import argparse
import csv
import json
from pathlib import Path


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    folder = Path(directory)
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    with (folder/'history.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('history.csv is empty')
    values = lambda key: [float(row[key]) for row in rows]
    time = values('time_s')
    pressure = values('prescribed_pressure_mmhg')
    tension = values('prescribed_tension_dyn_per_cm2')
    volume = values('cavity_volume_ml')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(time, pressure)
    axes[0, 0].set(xlabel='Time (s)', ylabel='Prescribed pressure (mmHg)')
    axes[0, 1].plot(time, tension)
    axes[0, 1].set(xlabel='Time (s)', ylabel='Prescribed active stress (dyn/cm2)')
    axes[1, 0].plot(time, volume)
    axes[1, 0].set(xlabel='Time (s)', ylabel='Cavity volume (mL)')
    axes[1, 1].plot(volume, pressure)
    axes[1, 1].scatter([volume[0], volume[-1]], [pressure[0], pressure[-1]], s=25)
    axes[1, 1].set(xlabel='Cavity volume (mL)', ylabel='Prescribed pressure (mmHg)')
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle(f'Ideal LV imposed-load demo: {report["status"]}, '
                 f't = {report["reached_time_s"]:.4f} s (period {report["period_s"]:g} s)\n'
                 'Pressure-volume trace uses prescribed traction, not a circulation model')
    target = folder/'cycle_curves.png'
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/lv_cycle')
    print(plot(parser.parse_args().input))

