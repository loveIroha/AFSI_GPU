"""Default ideal-LV entry: PyTorch Q2/Q1 FEM with native-node CUDA IB."""
if __package__:
    from .run_fem import main
else:
    from run_fem import main

if __name__=='__main__':
    main()
