# SelfGNNplus adapted from [SelfGNN](https://arxiv.org/abs/2405.20878)

Graph Neural Networks (GNNs) have demonstrated strong performance across various graph-structured applications, yet capturing temporal dependencies in recommendation systems remains a significant challenge. This study introduces SelfGNNplus, an enhanced model derived from SelfGNN, designed to incorporate interval-level dependencies and a self-supervised learning framework. SelfGNNplus improves upon its predecessor by better understanding and utilizing temporal patterns in user-item interactions. Comprehensive experiments across several benchmark datasets demonstrate that SelfGNNplus consistently outperforms state-of-the-art models, including Neural Collaborative Filtering (NCF), Transformers, and conventional GNNs. Notably, SelfGNNplus achieves up to 5.97% improvement in Hit Rate and 6.41% in NDCG compared to the best baseline models. Ablation studies highlight the critical role of interval-level dependencies, revealing their substantial impact on recommendation accuracy. The study also examines the effects of key hyperparameters on model performance and computational efficiency. Future work will explore advanced temporal encoding techniques, scalability improvements, and the integration of contextual information to further enhance recommendation precision and system adaptability.

## 📝 Environment

You can run the following command to download the codes faster:

```bash
git clone https://github.com/Sovik-Ghosh/SelfGNNplus.git
```

Then run the following commands:

```bash
pip install matplotlib
pip install numpy
pip install scipy
pip install tensorflow[and-cuda]
```

## 📚 Recommendation Dataset

We utilized four public datasets to evaluate: *Gowalla, MovieLens,Yelp* and *Amazon*. Following the common settings of implicit feedback, if user  has rated item , then the element  is set as 1, otherwise 0. We filtered out users and items with too few interactions.

We employ the most recent interaction as the test set, the penultimate interaction as the validation set, and the remaining interactions in the user behavior sequence as the training data.

The datasets are in the `./Dataset` folder:

```
- ./Dataset/amazon(yelp/movielens/gowalla)
|--- sequence    # user behavior sequences (List)
|--- test_dict    # test item for each users (Dict)
|--- trn_mat_time    # user-item graphs in different periods (sparse matrix)
|--- tst_int    # users to be test (List)
```

### Original Data

The original data of our dataset can be found from following links (thanks to their work):

- Yelp: https://www.yelp.com/dataset
- Amazon-book: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- Gowalla: [SNAP: Network datasets: Gowalla (stanford.edu)](https://snap.stanford.edu/data/loc-Gowalla.html)
- Movielens: [MovieLens 10M Dataset | GroupLens](https://grouplens.org/datasets/movielens/10m/)

### Methods for preprocessing original data

If you want to process your data into the several data files required for SA-GNN (i.e., `sequence`,`test_dict`,`trn_mat_time`,`tst_int`), you can refer to the following code for preprocessing the raw data of Amazon-book:

1. Download the original data file (for example, `amazon_ratings_Books.csv` from links for original datasets) and run [preprocess_to_trnmat.ipynb](./preprocess_to_trnmat.ipynb) to get the `trn_mat_time` and `tst_int` files, as well as other intermediate files (`train.csv`,`test.csv`).
2. Run [preprocess_to_sequece.ipynb](./preprocess_to_sequence.ipynb), which reads in the intermediate files (`train.csv` and `test.csv`) and finally generates the `sequence` and `test_dict` files.

You are welcome to modify the preprocessing code as needed to suit your data.

## 🚀 Examples to run the codes

You need to create the `./History/` and the `./Models/` directories. The command to train SelfGNNplus on the Gowalla/MovieLens/Amazon/Yelp dataset is as follows.

- Gowalla

```
sh gowalla.sh
```

- MovieLens

```
sh movielens.sh
```

- Amazon

```
sh amazon.sh
```

- Yelp

```
sh yelp.sh
```
