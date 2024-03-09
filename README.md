# Embedding Test



---

Set up Conda Environment

```sh
conda create --name embedtest python=3.9 -y
conda activate embedtest
pip install -r requirements.txt
```

Delete Conda Environment

```sh
conda deactivate
conda env remove --name embedtest
```

Getting the Dataset

The dataset is available at [here](https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval), and it is necessary to download it and place it in the `data` folder.

```sh
wget -O data/cisi.tar.gz https://www.kaggle.com/dmaso01dsta/cisi-a-dataset-for-information-retrieval/download
tar -xvf data/cisi.tar.gz -C data

```


---
