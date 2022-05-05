# Obtaining OpenMC cross sections

OpenMC reads cross section information from a set of hdf5 files, which can be obtained from various published libraries. This document describes how to obtain those libraries and set the required environment variable.

## Clone the OpenMC data package

The OpenMC team have made available a [data](https://github.com/openmc-dev/data) package that provides scripts for downloading and processing the cross section libraries. The data package should have been checked out when the openmc installer provided in BLUEPRINT was run, but if it isn't available then it can be obtained by running the below (assuming you're starting from the BLUEPRINT root directory):

```shell
cd ..
git clone https://github.com/openmc-dev/data.git
cd data
git checkout
```

## Download and process the cross section library

Once the OpenMC data package has been cloned you can use the scripts to download and process the required library. Note that this will download a very large compressed file (GBs in size) and will require additional space (10s of GBs) in order to process the library.

For example, the 2019 release of the tendl library can be obtained by running the below command in the directory containing the data package:

```shell
source ../BLUEPRINT/.env/bin/activate
python convert_tendl.py -r 2019 --cleanup
```

## Setting the OPENMC_CROSS_SECTIONS environment variable

Finally, the OPENMC_CROSS_SECTIONS environment variable should be set so that OpenMC is able to locate the cross section library. It is suggested to do this in the user's `~/.bashrc` file so that it is automatically set in each session.

The below commands give an example of how the OPENMC_CROSS_SECTIONS environment variable may be set. This assumes that the current working directory is the BLUEPRINT root directory and the data directory is at the level above the BLUEPRINT root.

```shell
cd ../data
export OPENMC_CROSS_SECTIONS=${PWD}/tendl-2019-hdf5/cross_sections.xml
```

## Troubleshooting

If the data directory was installed as root (via sudo), then the ownership of the data directory will need to be changed to the current user. This can be done by running the below command to change the ownership:

```shell
sudo chown -R $USER "data"
```
