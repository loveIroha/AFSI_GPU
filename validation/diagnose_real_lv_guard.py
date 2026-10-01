"""Inspect saved real-LV transport checks on CPU; no simulation or GPU setup."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from afsi_torch.transport import transport_numbers, transport_policy, transport_violations, implicit_policy


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
    numbers = transport_numbers(speeds.tolist(), spacing.tolist(), dt, mu/rho)
    implicit = settings.get('coupling', {}).get('scheme', 'explicit-lagged') == 'implicit-newton'
    violations = transport_violations(numbers['courant'], numbers['advection_diffusion_number'],numbers['viscous_number'])
    return dict(checkpoint=str(Path(checkpoint)), step=meta['step'], time_s=meta['time'],
                attempted_next_step=meta['step']+1, dt=dt, rho=rho, mu=mu,
                spacing_cm=spacing.tolist(), component_peaks=peaks,
                **numbers, transport_policy=implicit_policy() if implicit else transport_policy(),
                triggered=[] if implicit else violations,
                explicit_screen_triggered=violations,
                legacy_cell_re_gt_one=numbers['cell_reynolds'] > 1.,
                saved_failure=meta.get('progress', {}).get('failure'),
                inspection='selected saved fields; no full checksum validation or advancement',
                interpretation=('implicit transport diagnostics; explicit CFL/D/A vetoes do not apply; cell_Re monitors spatial resolution'
                    if implicit else 'current transport screen; cell_Re monitors spatial resolution, not time stability; saved failure may use an older policy'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.checkpoint), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
