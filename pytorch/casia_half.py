import os
from os.path import split, isdir, join
with open("CASIA-WebFace.list", 'r') as f:
  lines = f.readlines()
entities = []
for l in lines:
  entities.append(split(l.split()[0])[0])

entities = list(set(entities))

half = [i for (idx, i) in enumerate(entities) if idx % 2 == 0]

print(len(entities))
print(len(half))

source = "/media/data2/dataset/CASIA-WebFace"
dest = "/media/data2/dataset/CASIA-WebFace-half"

if not isdir(dest):
  os.makedirs(dest)

for i in half:
  os.symlink(join(source, i), join(dest, i))
  print("Making symlink from %s to %s" % (join(source, i), join(dest, i)))

