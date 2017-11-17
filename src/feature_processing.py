import logging
import os
from collections import defaultdict
from os import listdir
from os.path import isdir
from os.path import join
from random import seed
from random import shuffle

import numpy as np
from progressbar import ProgressBar
from python_speech_features import mfcc
from scipy.io.wavfile import read
from sklearn import decomposition

from context import Context


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
        self.lang_enc, _ = self.get_lang_maps()
        self.logger.info("lang encoder = %s" % self.lang_enc)

    def get_lang_maps(self):
        conf = self.context.conf
        decode = list(enumerate(os.listdir(conf.raw_dir)))
        encode = [(l,i) for (i,l) in decode]
        return dict(encode), dict(decode)

    def computeFeatures(self):
        conf = self.context.conf

        # Creating tuples (label, wav_file)
        self.logger.info('creating wav index tuples')
        path = conf.raw_dir
        if len(conf.langs) > 0:
            chose_lang = lambda x: isdir(join(path,x)) and x in conf.langs
        else:
            chose_lang = lambda x: isdir(join(path, x))
        audio_dirs = [x for x in listdir(path) if chose_lang(x)]
        audio_tuples = [ (x,join(path,x,y)) for x in audio_dirs for y in listdir(join(path, x)) ]
        self.logger.info("number of examples: " + str(len(audio_tuples)))
        [train_keys,cross_keys,test_keys] = self.split_dataset(audio_tuples)
        
        self.logger.info('computing training features')
        train_data = self.compute_mfccs(audio_tuples, train_keys, conf.train_dir)
        self.logger.info('computing cross-val features')
        cross_data = self.compute_mfccs(audio_tuples, cross_keys, conf.train_dir)
        self.logger.info('computing test features')
        test_data = self.compute_mfccs(audio_tuples, test_keys, conf.train_dir)

        print "Starting PCA for train, cross and test data..."

        pca = decomposition.PCA(n_components = conf.pca_decomposition)

        # Training dataset
        train_data_list = list(train_data)
        pca.fit(train_data_list[0])
        train_data_list[0] = pca.transform(train_data_list[0])
        train_data = tuple(train_data_list)

        # Cross dataset
        cross_data_list = list(cross_data)
        pca.fit(cross_data_list[0])
        cross_data_list[0] = pca.transform(cross_data_list[0])
        cross_data = tuple(cross_data_list)

        # Test dataset
        test_data_list = list(test_data)
        pca.fit(test_data_list[0])
        test_data_list[0] = pca.transform(test_data_list[0])
        test_data = tuple(test_data_list)

        print "Finished PCA"

        return train_data, cross_data, test_data
        
        #self.logger.info('computing cross validation features')
        #self.compute_mfccs(cross_keys, conf.cross_dir)
        
        #self.logger.info('computing test features')
        #self.compute_mfccs(test_keys, conf.test_dir)


    def writeLines(self, filename, lines):
        self.logger.debug("writing file: " + filename)
        fo = open(filename, "w+")
        fo.writelines(lines)
        fo.close()

    def split_dataset(self, audio_tuples):
        conf = self.context.conf
        seed(conf.rand_seed)
        audio_map = defaultdict(lambda: [])
        i = 0
        for (lan, file) in audio_tuples:
            audio_map[lan].append(i)
            i += 1
        train_keys, cross_keys, test_keys = [], [], []
        for lan, idxs in audio_map.iteritems():
            keys = list(idxs)
            shuffle(keys)
            n = len(keys)
            n_train = int(round(n*(conf.train_size/100.0)))
            n_cross = int(round(n*(conf.cross_size/100.0)))
            n_test = int(round(n*(conf.test_size/100.0)))
            self.logger.info(lan + ": number of training samples: " + str(n_train))
            self.logger.info(lan + ": number of cross validation samples: " + str(n_cross))
            self.logger.info(lan + ": number of testing samples: " + str(n_test))
            train_keys += keys[:n_train]
            cross_keys += keys[n_train:(n_train + n_cross)]
            test_keys += keys[(n_train + n_cross):(n_train + n_cross + n_test)]
        return (train_keys, cross_keys, test_keys)

    def compute_mfccs(self, audio_tuples, keys, path):
        if not keys:
            return None, None, []
        conf = self.context.conf
        bar = ProgressBar()
        X, Y, files = [], [], []
        n_cols = conf.mfcc_x_vec * conf.num_cep
        lang_cnt = defaultdict(lambda: 0)
        for key in bar(keys):
            # compute mfccs
            #print key, audio_tuples[key][1]
            audio_file = audio_tuples[key][1]
            rate,signal = read(audio_file)
            signal = np.cast[float](signal)
            #print rate, signal.shape, signal
            mfcc = self.computeMFCC(signal, rate)
            mfcc = ((mfcc-np.mean(mfcc))/np.std(mfcc))
            #print mfcc.shape, mfcc
            n_rows = mfcc.size / n_cols
            mfcc = mfcc.flatten()[:n_rows * n_cols]
            X += list(mfcc.reshape(n_rows, n_cols))
            lang = audio_tuples[key][0]
            y = self.lang_enc[lang]
            for _ in xrange(n_rows):
                Y.append(y)
                files.append(audio_file)
                lang_cnt[lang] += 1
        X = np.array(X)
        Y = np.array(Y)
        self.logger.info("X shape = %s, Y shape = %s, files len = %d" % \
                         (X.shape, Y.shape, len(files)))
        self.logger.info("lang cnt = %s, # diff files = %d" % \
                         (lang_cnt, len(set(files))))
        return X, Y, files
        
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
#    print features
#    print type(features)


