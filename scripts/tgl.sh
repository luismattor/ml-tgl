#!/bin/bash
#
# Entry point for the great language project
#
# title         ws.sh
# author        Luis Mateos
# date          05-06-2016
# usage         ./tgl.sh <task>
# notes

set -e

script_dir="$(dirname $(readlink -f $0))"
src_dir=$script_dir/../src
model_dir=$script_dir/../models
train_dir=$script_dir/../dataset/train
test_dir=$script_dir/../dataset/test
task=$1

export PYTHONIOENCODING=UTF-8
export WORD_SPOTTING_HOME="$script_dir/.."

function do_feature_processing() {
    /opt/anaconda/bin/python $src_dir/feature_processing.py
}

function do_training() {
    echo "Not yet"
}

function do_testing() {
    echo "Not yet"
}

case $task in
    features) do_feature_processing ;;
    train) do_training ;;
    test) do_testing ;;
    *) echo "Unknown task $task" ;;
esac
