# Third-party resources

The MIT license covers this project's original software only.

- DEAM audio and annotations: obtain them from https://cvml.unige.ch/databases/DEAM/ and follow its manual and each recording's applicable terms. This repository does not redistribute them or grant additional rights. Dataset source: Aljanaki, Yang and Soleymani, PLOS ONE 2017, DOI 10.1371/journal.pone.0173392.
- MERT-v1-95M weights and model code: https://huggingface.co/m-a-p/MERT-v1-95M. The runtime loads the specified revision, including its custom model implementation, under the upstream terms. It preserves the checkpoint's official feature-extractor preprocessing.
- Existing project code is adapted from TonyWang2182/Music-Arousal-Code, commit 0902550a82467158f5815bad0ee56ae78bd74109, under its MIT license; the copyright notice is retained.
- NumPy, SciPy, scikit-learn, PyTorch, Transformers, soundfile and other dependencies retain their respective upstream licenses.

Artist-disjoint downstream partitions do not establish that a pretrained model never encountered related recordings. No complete pretraining-overlap exclusion is claimed.
