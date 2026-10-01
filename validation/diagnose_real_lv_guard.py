"""Inspect saved real-LV transport checks on CPU; no simulation or GPU setup."""
import argparse
import json
from pathlib import Path
import numpy as np


def diagnose(checkpoint):
    # Diagnostic inspection only: load selected arrays without rebuilding the
    # model/graphs. Full checksum validation remains load_real_lv's job.
    with np.load(checkpoint, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        if (meta.get('producer') != 'afsi-torch-real-lv-mac'
                or meta.get('schema') != 1 or meta.get('units') != 'cm-g-s'):
            raise ValueError('a real-LV MAC checkpoint in cm-g-s units is required')
        settings = meta['settings']
        shape = tuple(settings['fluid_shape'])
        spacing = np.asarray(settings['fluid_lengths'], dtype=float)/shape
        origin = np.asarray(settings['fluid_origin'], dtype=float)
        dt, rho, mu = (float(settings[name]) for name in ('dt', 'rho', 'mu'))
        if (not np.isfinite([dt, rho, mu, *spacing, *origin]).all()
                or min(dt, rho, mu, *spacing) <= 0):
            raise ValueError('invalid saved fluid parameters')
        peaks = []
        for c in range(3):
            u = archive[f'velocity_{c}']
            expected = tuple(n+(c == axis) for axis, n in enumerate(shape))
            if u.shape != expected or not np.isfinite(u).all():
                raise ValueError('invalid saved MAC velocity')
            index = np.unravel_index(np.abs(u).argmax(), u.shape)
            offset = np.full(3, .5); offset[c] = 0.
            peaks.append(dict(component='xyz'[c], max_abs_cm_per_s=float(abs(u[index])),
                              value_cm_per_s=float(u[index]),
                              index=[int(i) for i in index],
                              location_cm=(origin+(np.asarray(index)+offset)*spacing).tolist(),
                              rms_cm_per_s=float(np.sqrt(np.mean(u*u)))))
    speeds = np.asarray([p['max_abs_cm_per_s'] for p in peaks])
    cfl = float(np.sum(dt*speeds/spacing))
    re = float(np.max(speeds*spacing/(mu/rho)))
    return dict(checkpoint=str(Path(checkpoint)), step=meta['step'], time_s=meta['time'],
                attempted_next_step=meta['step']+1, dt=dt, rho=rho, mu=mu,
                spacing_cm=spacing.tolist(), component_peaks=peaks,
                courant=cfl, courant_limit=.25, cell_reynolds=re, cell_reynolds_limit=1.,
                viscous_number=float(dt*mu/rho*np.sum(1/spacing**2)),
                triggered=[name for name, failed in
                           (('courant', cfl > .25), ('cell_reynolds', re > 1.)) if failed],
                component_velocity_at_cell_re_limit_cm_per_s=((mu/rho)/spacing).tolist(),
                saved_failure=meta.get('progress', {}).get('failure'),
                inspection='selected saved fields; no full checksum validation or advancement',
                interpretation='cell_Re is independent of dt; this conservative check alone does not prove divergence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.checkpoint), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
