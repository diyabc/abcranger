# ABC random forests for model choice and parameters estimation

[![Build Status](https://travis-ci.com/fradav/abcranger.svg?branch=master)](https://travis-ci.com/fradav/abcranger)

Methodologies based on :

- [@pudlo2015reliable]
- [@raynal2016abc]

Libraries we use :

- [Ranger : A Fast Implementation of Random Forests](https://github.com/imbs-hl/ranger) [@wright2015ranger], there tuned to make "online" calculations (not storing the whole forests in memory)
- [Eigen : C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.](http://eigen.tuxfamily.org)

## Model Choice

```text
 - ABC Random Forest/Model choice command line options
Usage:
  ModelChoice [OPTION...]

  -h, --header arg        Header file (default: headerRF.txt)
  -r, --reftable arg      Reftable file (default: reftableRF.bin)
  -b, --statobs arg       Statobs file (default: statobsRF.txt)
  -o, --output arg        Prefix output (default: modelchoice_out)
  -n, --nref arg          Number of samples, 0 means all (default: 0)
  -m, --minnodesize arg   Minimal node size. 0 means 1 for classification or
                          5 for regression (default: 0)
  -t, --ntree arg         Number of trees (default: 500)
  -j, --threads arg       Number of threads, 0 means all (default: 0)
  -s, --seed arg          Seed, 0 means generated (default: 0)
  -c, --noisecolumns arg  Number of noise columns (default: 5)
  -l, --lda               Enable LDA (default: true)
      --help              Print help
```

Header, reftable and statobs files should be in the current directory.

Three files are created :

- `modelchoice_out.ooberror` : OOB Error rate vs number of trees (line number is the number of trees)
- `modelchoice_out.importance` : variables importance (sorted)
- `modelchoice_out.predictions` : votes, prediction and posterior error rate
- `modelchoice_out.confusion` : OOB Confusion matrix of the classifier

## Parameter Estimation

```text
 - ABC Random Forest/Model parameter estimation command line options
Usage:
  ..\..\build\EstimParam.exe [OPTION...]

  -h, --header arg        Header file (default: headerRF.txt)
  -r, --reftable arg      Reftable file (default: reftableRF.bin)
  -b, --statobs arg       Statobs file (default: statobsRF.txt)
  -o, --output arg        Prefix output (default: estimparam_out)
  -n, --nref arg          Number of samples, 0 means all (default: 0)
  -m, --minnodesize arg   Minimal node size. 0 means 1 for classification or
                          5 for regression (default: 0)
  -t, --ntree arg         Number of trees (default: 500)
  -j, --threads arg       Number of threads, 0 means all (default: 0)
  -s, --seed arg          Seed, 0 means generated (default: 0)
  -c, --noisecolumns arg  Number of noise columns (default: 5)
  -p, --pls               Enable PLS (default: true)
      --chosenscen arg    Chosen scenario (mandatory)
      --ntrain arg        number of training samples (mandatory)
      --ntest arg         number of testing samples (mandatory)
      --parameter arg     name of the parameter of interest (mandatory)
      --help              Print help
```

Header, reftable and statobs files should be in the current directory.

Five files (or seven if pls activated) are created :

- `estimparam_out.ooberror` : OOB MSE rate vs number of trees (line number is the number of trees)
- `estimparam_out.importance` : variables importance (sorted)
- `estimparam_out.predictions` : expectation, variance and 0.05, 0.5, 0.95 quantile for prediction
- `estimparam_out.predweights` : weights of the prediction (for density plot)
- `estimparam_out.teststats` : various statistics on test (MSE, NMSE, NMAE etc.)

if pls enabled :

- `estimparam_out.plsvar` : variance explained by number of components
- `estimparam_out.plsweights` : variable weight in the first component (sorted by absolute value)

## TODO

### Input/Output

- [ ] Integrate hdf5 routines to save/load reftables/observed stats with associated metadata
  - [ ] Provide R code to save in hdf5
  - [ ]  Provide Python code to save in hdf5

### C++ standalone

- [ ] Merge the two methodologies in a single executable with options
- [ ] Mo ve to another options parser (CLI?)

### External interfaces

- [ ] R package
- [ ] Python package
  
### Documentation

- [ ] Code documentation
- [ ] Documentate the build

### Continuous integration

- [ ] Fix travis build. Currently the vcpkg download of eigen3 head is broken.
- [ ] osX travis build
- [ ] Appveyor win32 build

## Long/Mid term TODO

- Write our own tree/rf implementation with better storage efficiency than ranger
- Make functional tests for the two methodologies