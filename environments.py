import pickle
import time
import os
import json
import urllib

from PIL import Image
import numpy as np
import fasteners

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import flickrapi


class OpenImageClassifier():
    def __init__(self, class_id, initial_tag_size=None, seed=0):
        np.random.seed(seed)
        self.class_id = class_id

        with open('openimage_output.pickle', 'rb') as f:
            self.output = pickle.load(f)

        with open('openimage_image_to_tag.pickle', 'rb') as f:
            self.item_to_tag_dict = pickle.load(f)
        with open('openimage_tag_to_image.pickle', 'rb') as f:
            self.tag_to_item_dict = pickle.load(f)

        self.tags = list(self.tag_to_item_dict.keys())
        if initial_tag_size is not None:
            self.tags = np.random.choice(self.tags, size=initial_tag_size, replace=False).tolist()

    def item_to_tag(self, item):
        return self.item_to_tag_dict[item]

    def tag_to_item(self, tag):
        return self.tag_to_item_dict[tag]

    def f(self, item):
        return self.output[item][self.class_id]

    def get_image(self, item):
        return Image.open('imgs/' + item + '.jpg').convert('RGB')


class FlickerClassifier():
    def __init__(self, api_key, api_secret, class_id, seed=0):
        np.random.seed(seed)
        self.flickr = flickrapi.FlickrAPI(api_key, api_secret, format='json')

        self.item_to_tag_pickle = 'flickr_objects/cache_image_to_tag.pickle'
        self.tag_to_item_pickle = 'flickr_objects/cache_tag_to_image.pickle'
        self.item_to_url_pickle = 'flickr_objects/cache_image_to_url.pickle'
        self.results_pickle = 'flickr_objects/cache_results.pickle'
        self.initial_tags = 'flickr_objects/initial_tags.txt'
        self.cache_lock = 'flickr_objects/cache_lock'
        self.api_log = 'flickr_objects/api_log_{}'.format(api_key)
        self.api_lock = 'flickr_objects/api_lock_{}'.format(api_key)

        with fasteners.InterProcessLock(self.cache_lock):
            self.cache_item_to_tag = {}
            if os.path.exists(self.item_to_tag_pickle):
                with open(self.item_to_tag_pickle, 'rb') as f:
                    self.cache_item_to_tag = pickle.load(f)

            self.cache_tag_to_item = {}
            if os.path.exists(self.tag_to_item_pickle):
                with open(self.tag_to_item_pickle, 'rb') as f:
                    self.cache_tag_to_item = pickle.load(f)

            self.item_to_url = {}
            if os.path.exists(self.item_to_url_pickle):
                with open(self.item_to_url_pickle, 'rb') as f:
                    self.item_to_url = pickle.load(f)

            self.cache_results = {}
            if os.path.exists(self.results_pickle):
                with open(self.results_pickle, 'rb') as f:
                    self.cache_results = pickle.load(f)

        with open(self.initial_tags) as f:
            self.tags = [r.strip() for r in f]

        self.class_id = class_id

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = models.resnet18(pretrained=True)
        self.model.eval()

        if not os.path.exists('flickr_images'):
            os.makedirs('flickr_images')

    def wait_until_flickr_rate(self):
        with fasteners.InterProcessLock(self.api_lock):
            if os.path.exists(self.api_log):
                with open(self.api_log, 'r') as f:
                    times = f.readlines()
            else:
                times = []
            if len(times) == 3000:
                t = max(0, 3600 - (time.time() - float(times[0])))
                time.sleep(t)
                times.pop(0)
            times.append(time.time())
            assert(len(times) <= 3000)
            with open(self.api_log, 'w') as f:
                for r in times:
                    print(float(r), file=f)

    def item_to_tag(self, item):
        if item not in self.cache_item_to_tag:
            self.wait_until_flickr_rate()
            try:
                tags = self.flickr.tags.getListPhoto(photo_id=item)
                self.cache_item_to_tag[item] = [t['raw'] for t in json.loads(tags.decode('utf-8'))['photo']['tags']['tag']]
            except BaseException:
                self.cache_item_to_tag[item] = []
        return self.cache_item_to_tag[item]

    def tag_to_item(self, tag):
        if tag not in self.cache_tag_to_item:
            self.wait_until_flickr_rate()
            try:
                res = self.flickr.photos.search(tags=tag, license='9,10', extras='url_l', per_page=500)
                res = [r for r in json.loads(res.decode('utf-8'))['photos']['photo'] if 'url_l' in r and 'id' in r]
            except BaseException:
                res = []
            self.cache_tag_to_item[tag] = [i['id'] for i in res]
            for i in res:
                self.item_to_url[i['id']] = i['url_l']
        return self.cache_tag_to_item[tag]

    def get_image(self, item):
        filename = 'flickr_images/{}.jpg'.format(item)
        try:
            if not os.path.exists(filename):
                urllib.request.urlretrieve(self.item_to_url[item], filename)
            return Image.open(filename).convert('RGB')
        except BaseException:
            return Image.new('RGB', (256, 256))

    def f(self, item):
        key = (item, self.class_id)
        if key not in self.cache_results:
            input_item = self.get_image(item)
            input_tensor = self.preprocess(input_item)
            input_batch = input_tensor.unsqueeze(0)
            with torch.no_grad():
                self.cache_results[key] = self.model(input_batch).reshape(-1).numpy()[self.class_id]
        return self.cache_results[key]

    def merge_save(self, filename, dict):
        old_dict = {}
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                old_dict = pickle.load(f)
        for key, value in dict.items():
            old_dict[key] = value
        with open(filename, 'wb') as f:
            pickle.dump(old_dict, f)

    def save_cache(self):
        with fasteners.InterProcessLock(self.cache_lock):
            self.merge_save(self.item_to_tag_pickle, self.cache_item_to_tag)
            self.merge_save(self.tag_to_item_pickle, self.cache_tag_to_item)
            self.merge_save(self.item_to_url_pickle, self.item_to_url)
            self.merge_save(self.results_pickle, self.cache_results)


class FlickerSimilarity(FlickerClassifier):
    def __init__(self, api_key, api_secret, class_id, seed=0):
        super(FlickerSimilarity, self).__init__(api_key, api_secret, class_id, seed)

        modules = list(self.model.children())[:-1]
        self.extractor = nn.Sequential(*modules)

        input_item = Image.open('flickr_objects/source/{}'.format(class_id)).convert('RGB')
        input_tensor = self.preprocess(input_item)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            self.source_feature = self.extractor(input_batch).reshape(-1).numpy()

    def f(self, item):
        key = (item, self.class_id)
        if key not in self.cache_results:
            input_item = self.get_image(item)
            input_tensor = self.preprocess(input_item)
            input_batch = input_tensor.unsqueeze(0)
            with torch.no_grad():
                target_feature = self.extractor(input_batch).reshape(-1).numpy()
                self.cache_results[key] = np.exp(-np.linalg.norm(target_feature - self.source_feature) ** 2 / 100)
        return self.cache_results[key]


def get_class_ids(env_str):
    if env_str in ['open', 'flickr']:
        return [i * 100 for i in range(10)]
    elif env_str == 'flickrsim':
        return os.listdir('flickr_objects/source')
    assert(False)


def get_env(env_str, class_id, seed, api_key, api_secret):
    if env_str == 'open':
        return OpenImageClassifier(class_id=class_id, initial_tag_size=100, seed=seed)
    elif env_str == 'flickr':
        return FlickerClassifier(class_id=class_id, api_key=api_key, api_secret=api_secret, seed=seed)
    elif env_str == 'flickrsim':
        return FlickerSimilarity(class_id=class_id, api_key=api_key, api_secret=api_secret, seed=seed)
    assert(False)
