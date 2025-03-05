# https://gist.github.com/gmgunter/1b864e17767aeb1055f82c1d14c721b2
# User Guide's way
set -e
set -x # echo on

readonly HELP="usage: $0 [forward|historical] test_location

Run the SAS workflow on a dataset and compare the output against
a golden dataset.

positional arguments:
forward|historical  the mode to run the workflow in
test_location   the location to put the test data and run the workflow
"

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo "$HELP"
    exit 0
elif [[ "$#" -lt 2 ]]; then
    echo 'Illegal number of parameters' >&2
    echo "$HELP"
    exit 1
fi

mode="$1"
if [ "$mode" == "forward" ]; then
    echo "Running forward mode"
elif [ "$mode" == "historical" ]; then
    echo "Running historical mode"
else
    echo "Invalid mode: $mode"
    exit 1
fi

test_location="$2"
test_location=$(realpath $test_location)
mkdir -p $test_location

# Clone the source.
git clone git@github.com:opera-adt/disp-nisar.git
cd disp-nisar
git checkout v0.1

TAG=${TAG:-"$(whoami)/disp-nisar:0.1"}
# Build the docker image.
BASE="cae-artifactory.jpl.nasa.gov:16003/gov/nasa/jpl/iems/sds/infrastructure/base/jplsds-oraclelinux:8.4.230101"
./docker/build-docker-image.sh --tag "$TAG" --base "$BASE"

# untar the test data.
# Sizes of the tarballs:
# $ lsh *tar
# -rw-r--r-- 1 staniewi users 144G Oct  6 16:36 delivery_data_full.tar
# -rw-r--r-- 1 staniewi users 3.1G Oct  6 17:08 delivery_data_small.tar

tar -xf /home/smirzaee/dev/interface-delivery/delivery_data.tar -C $test_location
cd $test_location/delivery_data


# Run the SAS workflow.
# Pick the "historical" or "forward":
# $ ls config_files/run*
#     config_files/runconfig_forward.yaml  config_files/runconfig_historical.yaml

cfg_file="config_files/runconfig_${mode}.yaml"

docker run --rm -u $(id -u):$(id -g) \
    -v $PWD:/work \
    $TAG disp-nisar $cfg_file

# Compare the output against a golden dataset.
docker run \
    --rm \
    -v $PWD:/work \
    $TAG \
    disp-nisar validate golden_output/*nc  output/*nc