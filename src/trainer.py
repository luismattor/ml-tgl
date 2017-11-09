import logging
from collections import defaultdict

from sklearn import linear_model

from context import Context
from feature_processing import FeatureProcessing


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

    def check_file(self, f, files, Y, Yp, lang_dec):
        n = len(Y)
        lang_cnt = defaultdict(lambda: 0)
        exp_lang = set()
        for i in xrange(n):
            if f != files[i]:
                continue
            lang_cnt[Yp[i]] += 1
            exp_lang.add(Y[i])
        assert len(exp_lang) == 1
        exp_lang = lang_dec[exp_lang.pop()]
        pred_lang = lang_dec[max(lang_cnt, key=lambda l: lang_cnt[l])]
        if exp_lang == pred_lang:
            self.logger.info("failed for %s: lang_cnt = %s" % \
                             (f, sorted(lang_cnt.items(), key=lambda (l, c): -c)))
            return 0
        else:
            return 1

    def test(self, X, Y, files, lang_dec):
        self.logger.info("Testing model for X with shape %s" % \
                         str(X.shape))
        Yp = self.model.predict(X)
        cnt = 0
        err_cnt = 0.0
        for f in set(files):
            cnt += 1
            err_cnt += self.check_file(f, files, Y, Yp, lang_dec)
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
