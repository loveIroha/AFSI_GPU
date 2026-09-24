"""Optional scientific preview of output geometry (pip install -e '.[io]').

This reads an output artifact for plotting only; geometry generation itself
does not read mesh files. Cutaway hides triangles by centroid, not a CAD cut.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def preview(source, output):
    with np.load(source, allow_pickle=False) as data:
        X, faces, tags = data['X'], data['faces'], data['facet_tags']
        metadata = json.loads(str(data['metadata']))
    center = np.array(metadata['config']['center'])
    colors = {1: '#e5a153', 2: '#548fa8', 3: '#78ae82'}
    names = {1: 'ENDO', 2: 'EPI', 3: 'BASE'}
    polygons = X[faces[:, :3]]
    fig = plt.figure(figsize=(11, 6), layout='constrained')
    for column, cut in enumerate((False, True), 1):
        ax = fig.add_subplot(1, 2, column, projection='3d')
        keep = polygons.mean(1)[:, 1] >= center[1] if cut else np.ones(len(faces), dtype=bool)
        collection = Poly3DCollection(polygons[keep], facecolors=[colors[int(t)] for t in tags[keep]],
                                       edgecolors='#304651', linewidths=.35, alpha=1.)
        ax.add_collection3d(collection)
        lower, upper = X.min(0), X.max(0)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
               xlabel='x [cm]', ylabel='y [cm]', zlabel='z [cm]',
               title='Reference wall mesh' if not cut else 'Cutaway: base opening and inner wall')
        ax.set_box_aspect(upper-lower)
        ax.view_init(elev=23, azim=-65)
        ax.grid(False)
        ax.tick_params(labelsize=8)
    fig.suptitle('Generated ideal left ventricle | length: cm | volume: mL', fontsize=14)
    fig.legend(handles=[Patch(facecolor=colors[k], label=names[k]) for k in colors],
               loc='lower center', ncol=3, frameon=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/ideal_lv/generated.npz')
    parser.add_argument('--output', default='results/ideal_lv/preview.png')
    args = parser.parse_args()
    preview(args.source, args.output)
