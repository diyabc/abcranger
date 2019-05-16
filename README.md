  - [Model Choice](#model-choice)
  - [Parameter Estimation](#parameter-estimation)
  - [TODO](#todo)
  - [References](#references)

<!-- pandoc -f markdown README-ORIG.md -t gfm -o README.md --bibliography=ref.bib -s --toc --toc-depth=1 -->

[![Build
Status](https://travis-ci.com/fradav/abcranger.svg)](https://travis-ci.com/fradav/abcranger)

Random forests methodologies for :

  - ABC model choice (Pudlo et al. [2015](#ref-pudlo2015reliable))
  - ABC Bayesian parameter inference (Raynal et al.
    [2016](#ref-raynal2016abc))

Libraries we use :

  - [Ranger](https://github.com/imbs-hl/ranger) (Wright and Ziegler
    [2015](#ref-wright2015ranger)) : we use our own fork and have tuned
    forests to do “online”\[1\] computations (Growing trees AND making
    predictions at the same time, in order to avoid to store the whole
    forest in memory)\[2\].
  - [Eigen3](http://eigen.tuxfamily.org) (Guennebaud, Jacob, and others
    [2010](#ref-eigenweb))

As a mention, we use our own implementation of LDA and PLS from
(Friedman, Hastie, and Tibshirani [2001](#ref-friedman2001elements)).

There is two sets of binaries, one for model choice `ModelChoice`,
another for parameter estimation `EstimParam`. Each set contains a
Macos/Linux/Windows (x64 only) binary for each platform. There are
available within the
“[Releases](https://github.com/fradav/abcranger/releases)” tab, under
“Assets” section (unfold it to see the list).

Those are pure command line binaries, and they are no prerequisites or
library dependencies in order to run them. Just download them and launch
them from your terminal software of choice. The usual caveats with
command line executable apply there : if you’re not proficient with the
command line interface of your platform, please learn some basics or ask
someone who might help you in those matters.

As a note, we may add a graphical interface in a near future.

# Model Choice

## Usage

``` text
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

## Example

Example :

`ModelChoice -t 10000 -j 8`

Header, reftable and statobs files should be in the current directory.

## Generated files

Four files are created :

  - `modelchoice_out.ooberror` : OOB Error rate vs number of trees (line
    number is the number of trees)
  - `modelchoice_out.importance` : variables importance (sorted)
  - `modelchoice_out.predictions` : votes, prediction and posterior
    error rate
  - `modelchoice_out.confusion` : OOB Confusion matrix of the classifier

# Parameter Estimation

## Usage

``` text
 - ABC Random Forest/Model parameter estimation command line options
Usage:
  EstimParam [OPTION...]

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

## Example

Example (working with the dataset in `test/data`) :

`EstimParam -t 1000 -j 8 --parameter ra --chosenscen 1 --ntrain 1000
--ntest 50`

Header, reftable and statobs files should be in the current directory.

## Generated files

Five files (or seven if pls activated) are created :

  - `estimparam_out.ooberror` : OOB MSE rate vs number of trees (line
    number is the number of trees)
  - `estimparam_out.importance` : variables importance (sorted)
  - `estimparam_out.predictions` : expectation, variance and 0.05, 0.5,
    0.95 quantile for prediction
  - `estimparam_out.predweights` : csv of the value/weights pairs of the
    prediction (for density plot)
  - `estimparam_out.teststats` : various statistics on test (MSE, NMSE,
    NMAE etc.)

if pls enabled :

  - `estimparam_out.plsvar` : variance explained by number of components
  - `estimparam_out.plsweights` : variable weight in the first component
    (sorted by absolute value)

# TODO

## Input/Output

  - [ ] Integrate hdf5 (or msgpack?) routines to save/load
    reftables/observed stats with associated metadata
      - [ ] Provide R code to save/load the data
      - [ ] Provide Python code to save/load the data

## C++ standalone

  - [ ] Merge the two methodologies in a single executable with options
  - [ ] Move to another options parser (CLI?)

## External interfaces

  - [ ] R package
  - [ ] Python package

## Documentation

  - [ ] Code documentation
  - [ ] Documentate the build

## Continuous integration

  - [ ] Fix travis build. Currently the vcpkg download of eigen3 head is
    broken.
  - [ ] osX travis build
  - [ ] Appveyor win32 build

## Long/Mid term TODO

  - methodologies parameters auto-tuning
      - auto-discovering the optimal number of trees by monitoring OOB
        error
      - auto-limiting number of threads by available memory
  - Streamline the two methodologies (model choice and then parameters
    estimation)
  - Write our own tree/rf implementation with better storage efficiency
    than ranger
  - Make functional tests for the two methodologies

# References

<div id="refs" class="references">

<div id="ref-friedman2001elements">

Friedman, Jerome, Trevor Hastie, and Robert Tibshirani. 2001. *The
Elements of Statistical Learning*. Vol. 1. 10. Springer series in
statistics New York, NY, USA:

</div>

<div id="ref-eigenweb">

Guennebaud, Gaël, Benoît Jacob, and others. 2010. “Eigen V3.”
http://eigen.tuxfamily.org.

</div>

<div id="ref-pudlo2015reliable">

Pudlo, Pierre, Jean-Michel Marin, Arnaud Estoup, Jean-Marie Cornuet,
Mathieu Gautier, and Christian P Robert. 2015. “Reliable Abc Model
Choice via Random Forests.” *Bioinformatics* 32 (6): 859–66.

</div>

<div id="ref-raynal2016abc">

Raynal, Louis, Jean-Michel Marin, Pierre Pudlo, Mathieu Ribatet,
Christian P Robert, and Arnaud Estoup. 2016. “ABC Random Forests for
Bayesian Parameter Inference.” *arXiv Preprint arXiv:1605.05537*.

</div>

<div id="ref-wright2015ranger">

Wright, Marvin N, and Andreas Ziegler. 2015. “Ranger: A Fast
Implementation of Random Forests for High Dimensional Data in C++ and
R.” *arXiv Preprint arXiv:1508.04409*.

</div>

</div>

1.  The term “online” there and in the code has not the usual meaning it
    has, as coined in “online machine learning”. We still need the
    entire training data set at once. Our implementation is an “online”
    one not by the sequential order of the input data, but by the
    sequential order of computation of the trees in random forests,
    which are discarded once computed.

2.  We only use the C++ Core of ranger, which is under [MIT
    License](https://raw.githubusercontent.com/imbs-hl/ranger/master/cpp_version/COPYING),
    same as ours.
