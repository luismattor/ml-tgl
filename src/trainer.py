import logging
from collections import defaultdict

from feature_processing import FeatureProcessing
from itertools import izip


class Trainer:
    def __init__(self, context, model):
        self.context = context
        self.conf = self.context.conf
        self.logger = logging.getLogger('root')
        self.model = model

    def train(self, X, Y):
        self.logger.info("Training model for X with shape %s" % \
                         str(X.shape))
        self.model.fit(X, Y)

    def count_preds(self, Y, Yp, files, lang_dec):
        cnt = 0
        lang_cnt = dict()
        exp_lang = dict()
        for y, yp, f in izip(Y, Yp, files):
            if f not in lang_cnt:
                lang_cnt[f] = defaultdict(lambda: 0)
            lang_cnt[f][lang_dec[yp]] += 1
            if f not in exp_lang:
                exp_lang[f] = lang_dec[y]
                cnt += 1
        return cnt, lang_cnt, exp_lang

    def test(self, X, Y, files, lang_dec):
        self.logger.info("Testing model for X with shape %s" % \
                         str(X.shape))
        Yp = self.model.predict(X)
        self.logger.info("Computing error rate")
        err_cnt = 0.0
        cnt, lang_cnt, exp_lang = self.count_preds(Y, Yp, files, lang_dec)
        for f, lang in exp_lang.iteritems():
            pred_lang = max(lang_cnt[f], key=lambda l: lang_cnt[f][l])
            if pred_lang != lang:
                err_cnt += 1
                #print("failed for %s: %s != %s, %s" % \
                #      (f, pred_lang, lang, lang_cnt[f]))
        self.logger.info("error rate = %d/%d = %f" % (err_cnt, cnt, err_cnt / cnt))

    def train_and_test(self):
        fp = FeatureProcessing(self.context)
        _, lang_dec = fp.get_lang_maps()
        train_data, cross_data, test_data = fp.computeFeatures()
        X, Y, files = train_data
        self.train(X, Y)
        self.test(X, Y, files, lang_dec)
        X, Y, files = cross_data
        if files:
            self.test(X, Y, files, lang_dec)
        X, Y, files = test_data
        if files:
            self.test(X, Y, files, lang_dec)
