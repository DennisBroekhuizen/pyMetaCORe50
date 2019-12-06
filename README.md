# pyMeta CORe50

This repository provides an extension to the [pyMeta](https://github.com/spiglerg/pyMeta) library, which can be used to run meta-learning experiments with the [CORe50](https://github.com/vlomonaco/core50) continual learning dataset. In this example, training is done via Google Colab.

## Usage

1. [Download](https://vlomonaco.github.io/core50/index.html#download) the dataset: core50_imgs.npz & paths.pkl
2. [Download](https://github.com/spiglerg/pyMeta/archive/master.zip) pyMeta
3. Unzip and upload pyMeta to Google Drive
4. Inside the folder 'pyMeta-master/datasets' create a new folder called 'core50'
5. Upload the files core50_imgs.npz & paths.pkl to the folder 'pyMeta-master/datasets/core50'
6. [Download](https://github.com/DennisBroekhuizen/pyMetaCORe50/archive/master.zip) this repository
7. Upload the folder 'core50' of this repository to 'pyMeta-master/pyMeta'
8. Upload the file 'core50_metatrain.py' of this repository to the root folder 'pyMeta-master'
9. Upload the file 'COReTrain.ipynb' to Google Drive and open the file in Google Colab
10. In Google Colab make sure to change runtime type to GPU

Happy training!

## License
[MIT](https://choosealicense.com/licenses/mit/)
