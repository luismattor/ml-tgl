from glob import glob
from os import listdir
from os.path import isfile
from os.path import join
from progressbar import ProgressBar
from python_speech_features import mfcc
from random import shuffle
from random import seed
from scipy.io.wavfile import read
from context import Context

import fileinput
import logging
import numpy
import os
import sys

# TODO Expand dataset, split audio files in X seconds?
# TODO Transform labels to a int[] vector
# TODO Try multiclass logistic regression (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
class FeatureProcessing:

    target_sep = " "
    feats_prefix = "feats"
    feats_suffix = ".npy"
    alis_prefix = "alis"
    alis_suffix = ".txt"
    keys_prefix = "keys"
    keys_suffix = ".txt"
    wav_dir = "wav"
    txt_dir = "etc"
    mfc_str = "/mfc/"

    def __init__(self, context):
        self.context = context
        self.logger = logging.getLogger('root')

    def computeFeatures(self):
        conf = self.context.conf

        # Creating tuples (label, wav_file)
        self.logger.info('creating wav index tuples')
        path = conf.raw_dir
        audio_dirs = [x for x in listdir(path) if not isfile(join(path,x))]
        audio_tuples = [ (x,join(path,x,y)) for x in audio_dirs for y in listdir(join(path, x)) ]
        self.logger.info("number of examples: " + str(len(audio_tuples)))
        for x in audio_tuples:
            print x

        [train_keys,cross_keys,test_keys] = self.split_dataset(audio_tuples)
        
        self.logger.info('computing trainging features')
        train_mfccs = self.compute_mfccs(audio_tuples, train_keys, conf.train_dir)

        return train_mfccs
        
        #self.logger.info('computing cross validation features')
        #self.compute_mfccs(cross_keys, conf.cross_dir)
        
        #self.logger.info('computing test features')
        #self.compute_mfccs(test_keys, conf.test_dir)


    def writeLines(self, filename, lines):
        self.logger.debug("writing file: " + filename)
        fo = open(filename, "w+")
        fo.writelines(lines)
        fo.close()

    # TODO
    # - Ensure training set has samples from all languages
    # - apply dataset split in a language level-basis
    def split_dataset(self, audio_tuples):
        conf = self.context.conf
        keys = range(len(audio_tuples))
        seed(301214)
        shuffle(keys)
        n = len(keys)
        n_train = int(round(n*(conf.train_size/100.0)))
        n_cross = int(round(n*(conf.cross_size/100.0)))
        n_test = n - (n_train + n_cross)

        self.logger.info("number of training samples: " + str(n_train))
        self.logger.info("number of cross validation samples: " + str(n_cross))
        self.logger.info("number of testing samples: " + str(n_test))
        return (keys[:n_train],
                keys[n_train:(n_train + n_cross)],
                keys[(n_train + n_cross):])

    def compute_mfccs(self, audio_tuples, keys, path):
        conf = self.context.conf
        bar = ProgressBar()

        mfccs = []
        for key in bar(keys):
            # compute mfccs
            print key, audio_tuples[key][1]
            rate,signal = read(audio_tuples[key][1])
            signal = numpy.cast[float](signal)
            print rate, signal.shape, signal
            mfcc = self.computeMFCC(signal, rate)
            mfcc = ((mfcc-numpy.mean(mfcc))/numpy.std(mfcc)).flatten()
            print mfcc.shape, mfcc
            mfccs.append((audio_tuples[key][0], mfcc))
        
        return mfccs
        
    def computeMFCC(self, signal, rate):
        conf = self.context.conf
        return mfcc(signal, rate,
                 conf.win_len,
                 conf.win_step,
                 conf.num_cep,
                 conf.n_filt,
                 conf.n_fft,
                 conf.low_freq,
                 conf.high_freq,
                 conf.pre_emph,
                 conf.cep_lifter,
                 conf.append_energy)

if __name__ == "__main__":
    context = Context()
    fp = FeatureProcessing(context)
    features = fp.computeFeatures()
    print features
    print type(features)


