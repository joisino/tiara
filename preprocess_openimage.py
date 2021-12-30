import pickle
import numpy as np

with open('test-annotations-human-imagelabels.csv') as f:
    f.readline()
    images = [r.split(',')[0] for r in f]

images = sorted(list(set(images)))

np.random.seed(0)
res = np.random.choice(images, size=10000, replace=False)

with open('openimage_id.txt', 'w') as f:
    for i in res:
        print('test/{}'.format(i), file=f)

ss = set(res)

tag_to_name = {}
with open('oidv6-class-descriptions.csv') as f:
    for r in f:
        split = r.split(',')
        tag = split[0]
        name = ','.join(split[1:])
        tag_to_name[tag] = name.strip()

image_to_tag = {i: [] for i in res}
tag_to_image = {}

with open('test-annotations-human-imagelabels.csv') as f:
    f.readline()
    for r in f:
        imageid, source, tag, confidence = r.split(',')
        tag = tag_to_name[tag]
        if imageid in ss and int(confidence) == 1:
            image_to_tag[imageid].append(tag)
            if tag not in tag_to_image:
                tag_to_image[tag] = []
            tag_to_image[tag].append(imageid)

with open('openimage_image_to_tag.pickle', 'wb') as f:
    pickle.dump(image_to_tag, f)

with open('openimage_tag_to_image.pickle', 'wb') as f:
    pickle.dump(tag_to_image, f)
