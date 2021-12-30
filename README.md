# Retrieving Black-box Optimal Images from External Databases (WSDM 2022)

We propose how a user retreives an optimal image from external databases of web services (e.g., Flickr) with respect to user-defined functions (e.g., deep learning-based score functions.)

## 💿 Dependency

Please install

* `wget` and `unzip`, e.g., by `sudo apt install wget unzip`,
* [PyTorch](https://pytorch.org/) from the [official website](https://pytorch.org/), and
* other dependencies by `pip install -r requirements.txt`.

## 📂 Files

* `download.sh` downloads and preprocesses the Open Image dataset.
* `environments.py` implements wrappers of APIs, i.e., the oracles in the paper.
* `evaluate.py` is the evaluation script.
* `methods.py` implements Tiara, Tiara-S, and baseline methods.
* `openimage_feature_extract.py` preprocess the Open Image dataset. Please run this script *after* you download images. This script is automatically run by `download.sh`.
* `preprocess_openimage.py` preprocess the Open Image dataset. Please run this script *before* you download images. This script is automatically run by `download.sh`.
* `utils.py` implements miscellaneous functions, i.e., the word embbeding loader. 

## 🗃️ Download and Preprocess Datasets

```
$ bash ./download.sh
```

## 🧪 Evaluation

Try with Open Image datasets by

```
$ python evaluate.py --env open --verbose --num_seeds 1 -c 0
```

The results are saved in `outputs` directiory.

Please refer to the help command for further options.

```
$ python evaluate.py -h
usage: evaluate.py [-h] [--tuning] [--extra] [--env {open,flickr,flickrsim}]
                   [--num_seeds NUM_SEEDS] [--budget BUDGET]
                   [--api_key API_KEY] [--api_secret API_SECRET]
                   [--font_path FONT_PATH] [--verbose]
                   [-c [CLASSES [CLASSES ...]]]

optional arguments:
  -h, --help            show this help message and exit
  --tuning
  --extra
  --env {open,flickr,flickrsim}
  --num_seeds NUM_SEEDS
  --budget BUDGET
  --api_key API_KEY     API key for Flickr.
  --api_secret API_SECRET
                        API secret key for Flickr.
  --font_path FONT_PATH
                        Font path for wordclouds.
  --verbose
  -c [CLASSES [CLASSES ...]], --classes [CLASSES [CLASSES ...]]
```

### Flickr API

The Flickr experiments require a Flickr API key. Please get a key from [Flickr official website](https://www.flickr.com/services/apps/create/).

## 🖋️ Citation

```
@inproceedings{sato2022retrieving,
  author    = {Ryoma Sato},
  title     = {Retrieving Black-box Optimal Images from External Databases},
  booktitle = {Proceedings of the Fifteenth {ACM} International Conference on Web Search and Data Mining, {WSDM}},
  year      = {2022},
}
```