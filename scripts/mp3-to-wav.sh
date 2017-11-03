#!/bin/bash
#
# Utility functions for converting audio files into wav format
#
# title                 mp3-to-wav.sh
# author                Luis Mateos
# date                  03-10-2017
# usage                 ./mp3-to-wav.sh
# notes


set -e

scriptdir="$(dirname $(readlink -f $0))"
mp3dir="$scriptdir/../dataset/mp3"
wavdir="$scriptdir/../dataset/wav"

# pcm_s16le
#   PCM         traditional wave like format (raw bytes, basically)
#   s           signed
#   16          16 bits per sample
#   le          little endian
function convert_to_wav() {
    mkdir -p $wavdir
    pushd $mp3dir > /dev/null
        for dir in ./*; do
            mkdir -p "$wavdir/$dir"
            pushd $mp3dir/$dir > /dev/null
                for file in *.mp3; do
                    ffmpeg -i "$file" -acodec pcm_s16le -ac 1 -ar 16000 $wavdir/$dir/"${file%.mp3}.wav";
                done
            popd > /dev/null
        done
    popd > /dev/null
}

echo "$(date) Converting mp3 files in $mp3dir to wav in $wavdir"
convert_to_wav
