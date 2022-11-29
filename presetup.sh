#!/bin/bash
tee > /tmp/oneAPI.repo << EOF
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
if [ -x "$(command -v yum-config-manager)" ]; then
    mv /tmp/oneAPI.repo /etc/yum.repos.d
    yum install -y intel-oneapi-mkl-devel-2022.2.1 zip
fi
# cd /tmp
# apt-get update
# apt-get install -y wget software-properties-common libssl-dev apt-utils apt-transport-https ca-certificates curl zip unzip tar dirmngr
# apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F
# echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic main" >> /etc/apt/sources.list
# wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
# apt-get update
# apt-get install -y intel-mkl-64bit-2020.0-088 gcc-9 g++-9