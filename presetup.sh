#!/bin/bash
if [ -x "$(command -v yum-config-manager)" ]; then
    yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
    yum install -y intel-mkl-64bit-2020.0-088
    yum install -y curl zip unzip tar
fi