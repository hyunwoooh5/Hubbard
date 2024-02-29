# Hubbard Model for contour deformation


## Workflow

First define a physical model with a text file. For example, with improved Gaussian action, the following is saved as model.dat:
```
hubbard.ImprovedGaussianModel(
L=4,
nt=16,
Kappa=1,
U=8,
Mu=1.5,
dt=1/16
)
```

Then use `contour.py` to find a contour to maximize the average sign. After one finds a contour, generate configurations with the given contour using `sample.py`. Finally, one can estimate observables with errors through `bootstrap.py`.

Here is an example:
```
./contour.py model.dat c.pickle -l 1 -w 1 -lr 1e-3 # Terminate with CTRL-C
./sample.py model.dat c.pickle -N 1000 > sam.dat \&
./bootstrap.py < sam.dat
```