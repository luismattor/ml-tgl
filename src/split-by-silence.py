import logging
import os
from os.path import join
import shutil

from pydub import AudioSegment
from pydub.silence import split_on_silence

from context import Context


class SplitBySilence:
    def __init__(self, context):
        self.context = context
        self.conf = self.context.conf
        self.in_dir = self.conf.raw_dir
        self.out_dir = self.conf.split_by_silence_dir
        self.logger = logging.getLogger('root')

    def split_lang(self, lang):
        in_dir = join(self.in_dir, lang)
        out_dir = join(self.out_dir, lang)
        if os.path.exists(out_dir):
            self.logger.info("Deleting previous contents of %s" % out_dir)
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        for wav_file in os.listdir(in_dir):
            if not wav_file.endswith('.wav'):
                self.logger.warn("Ignoring non wav file %s" % wav_file)
                continue
            wav_path = join(in_dir, wav_file)
            sound_file = AudioSegment.from_wav(wav_path)
            audio_chunks = split_on_silence(sound_file,
                                            min_silence_len=500,
                                            silence_thresh=sound_file.dBFS-3
                                            )
            wav_pfx = wav_file.split(".")[0]
            for i, chunk in enumerate(audio_chunks):
                out_file = join(out_dir, "%s-c%04d.wav" % (wav_pfx, i))
                chunk.export(out_file, format="wav")
            self.logger.info("Split %s into %d chunks" % (wav_pfx, len(audio_chunks)))

    def split_all(self):
        for lang in os.listdir(self.in_dir):
            self.split_lang(lang)


if __name__ == "__main__":
    splitter = SplitBySilence(Context())
    splitter.split_all()
