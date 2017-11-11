# Given a csv file whose fst column is file, and snd
# column is associated label; move the files into
# directories named after the label.

import os
import sys
import shutil

assert os.path.isfile(sys.argv[1]), 'missing csv file arg'
for line in open(sys.argv[1], "r"):
    f, label = map(lambda s: s.strip(), line.split(","))
    if not os.path.exists(f):
        continue
    print("moving %s into %s" % (f, label))
    if not os.path.exists(label):
        os.makedirs(label)
    shutil.move(f, label)

