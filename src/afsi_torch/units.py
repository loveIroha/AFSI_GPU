"""Explicit CGS convention for physical examples; kernels remain unit-agnostic.

cm, g, s -> force dyn, stress dyn/cm^2, energy erg. 1 cm^3 = 1 mL.
AFSI demo_337 uses this mmHg conversion. This module does not rescale tensors.
"""
MMHG_TO_DYN_PER_CM2 = 1333.22368421
CGS_UNITS = dict(length='cm', volume='cm^3 (mL)', time='s', mass='g',
                 force='dyn', stress='dyn/cm^2', pressure='dyn/cm^2',
                 energy='erg', spring_coefficient='dyn/cm^3')
