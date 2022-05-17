#!/bin/bash
# if [ -x "$(command -v yum-config-manager)" ]; then
#     yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
#     rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
#     yum install -y intel-mkl-64bit-2020.0-088
 
# fi
cd /tmp
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
apt-get update
apt-get install -y intel-mkl-64bit-2020.0-088 openssl