# ABC random forests for model choice and parameters estimation

[![Build Status](https://travis-ci.com/fradav/abcranger.svg?branch=master)](https://travis-ci.com/fradav/abcranger)

Methodologies based on :

- [@pudlo2015reliable]
- [@raynal2016abc]

Libraries we use :

- [Ranger : A Fast Implementation of Random Forests](https://github.com/imbs-hl/ranger) [@wright2015ranger], there tuned to make "online" calculations (not storing the whole forests in memory)
- [Eigen : C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.](http://eigen.tuxfamily.org)

# Usage
```
 - ABC Random Forest/Model choice command line options
Usage:
  ModelChoice [OPTION...]

  -h, --header arg        Header file (default: headerRF.txt)
  -r, --reftable arg      Reftable file (default: reftableRF.bin)
  -b, --statobs arg       Statobs file (default: statobsRF.txt)
  -o, --output arg        Prefix output (default: onlineranger_out)
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
- `onlineranger_out.ooberror` : OOB Error rate vs number of trees (line number is the number of trees)
- `onlineranger_out.confusion` : OOB Confusion matrix of the classifier
- `onlineranger_out.importance` : variables importance (sorted)

