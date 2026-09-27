"""AFSI demo_337 pressure/active contraction ramp (1.5 s), held until 2 s."""
if __package__:
    from .lv_cycle import main
else:
    from lv_cycle import main

if __name__ == '__main__':
    main(default_profile='afsi337')

