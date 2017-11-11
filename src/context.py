from os.path import join

import codecs
import logging
import logging.config
import os
import ConfigParser

class Context:

    cfg_dir = "config"
    log_dir = "log"

    def __init__(self):
        self.home = os.environ['WORD_SPOTTING_HOME']
        self.start_logging_service()
        self.start_configuration_service()
        self.log_configuration()
    
    def start_logging_service(self):
        cfgFile = join(self.home, self.cfg_dir, "logging.conf")
        logFile = join(self.home, self.log_dir, "word-spotting.log")
        logging.config.fileConfig(cfgFile, defaults={"logfilename": logFile})

    def start_configuration_service(self):
        cfg_file = join(self.home, self.cfg_dir, "config.cfg")
        self.conf = Conf(cfg_file)

    def log_configuration(self):
        conf = self.conf
        logger = logging.getLogger("root")

        logger.info("Project home: " + self.home)

        logger.info("MFCC parameters")
        logger.info("*win_len: " + str(conf.win_len))
        logger.info("*win_step: " + str(conf.win_step))
        logger.info("*num_cep: " + str(conf.num_cep))
        logger.info("*n_filt: " + str(conf.n_filt))
        logger.info("*n_fft: " + str(conf.n_fft))        
        logger.info("*low_freq: " + str(conf.low_freq))
        logger.info("*high_freq: " + str(conf.high_freq))
        logger.info("*pre_emph: " + str(conf.pre_emph))
        logger.info("*cep_lifter: " + str(conf.cep_lifter))
        logger.info("*append_energy: " + str(conf.append_energy))
        
        logger.info("Dataset parameters")
        logger.info("*dataset_dir: " + conf.dataset_dir)
        logger.info("*raw_dir: " + conf.raw_dir)
        logger.info("*train_dir: " + conf.train_dir)
        logger.info("*cross_dir: " + conf.cross_dir)
        logger.info("*test_dir: " + conf.test_dir)
        logger.info("*train_size: " + str(conf.train_size))
        logger.info("*cross_size: " + str(conf.cross_size))
        logger.info("*test_size: " + str(conf.test_size))
        logger.info("*id_prefix: " + str(conf.id_prefix))
        logger.info("*batch_size: " + str(conf.batch_size))
        
        logger.info("Session and summary parameters")
        
        logger.info("Model parameters")

class Conf:

    TRUE = "True"
    
    def __init__(self, cfg_file):
        self.home = os.environ['WORD_SPOTTING_HOME']
        cfg_file = codecs.open(cfg_file, "r", encoding="utf-8")
        config = ConfigParser.SafeConfigParser()
        config.readfp(cfg_file)

        # MFCC Configuration
        self.win_len = float(config.get("mfcc", "win_len"))
        self.win_step = float(config.get("mfcc", "win_step"))
        self.num_cep = int(config.get("mfcc", "num_cep"))
        self.n_filt = int(config.get("mfcc", "n_filt"))
        self.n_fft = int(config.get("mfcc", "n_fft"))
        self.low_freq = int(config.get("mfcc", "low_freq"))
        self.high_freq = int(config.get("mfcc", "high_freq"))
        self.pre_emph = float(config.get("mfcc", "pre_emph"))
        self.cep_lifter = int(config.get("mfcc", "cep_lifter"))
        self.append_energy = \
                config.get("mfcc", "append_energy").strip() == self.TRUE

        # Split dataset configuration
        self.train_size = float(config.get("dataset", "train"))
        self.cross_size = float(config.get("dataset", "cross"))
        self.test_size = float(config.get("dataset", "test"))
        # TODO: Assert sizes sum to one
        self.id_prefix = config.get("dataset", "id_prefix").strip()
        self.mfcc_x_vec = int(config.get("dataset", "mfcc_x_vec").strip())
        self.langs = config.get("dataset", "langs").strip()
        self.langs = set([s for s in self.langs.split(',') if s])
        self.rand_seed = config.get("dataset", "rand_seed").strip()
        if self.rand_seed:
           self.rand_seed = int(self.rand_seed)
        else:
            self.rand_seed = None

        # Dataset folders
        self.dataset_dir = join(self.home, config.get("app","dataset_dir"))
        self.raw_dir = join(self.dataset_dir, config.get("app","raw_dir"))
        self.split_by_silence_dir = \
            join(self.dataset_dir, config.get("app", "split_by_silence_dir"))
        self.train_dir = join(self.dataset_dir, config.get("app","train_dir"))
        self.cross_dir = join(self.dataset_dir, config.get("app","cross_dir"))
        self.test_dir = join(self.dataset_dir, config.get("app","test_dir"))
        self.batch_size = int(config.get("app","batch_size"))
