# Recommendation System

There are two solutions presented in the repository:

* [Wide and Deep Neural Network](https://arxiv.org/abs/1606.07792)
* SVM (Final Model)

### Author

* Nguyen Gia Trong
* BS20-AI
* g.nguyen@innopolis.university

## Dataset

Used dataset is [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).

## Evaluation
To evaluate final model (SVD) on  u1.base, u1.test, u2.base, u2.test, u3.base, u3.test, u4.base, u3.test, u5.base,
u5.test, ua.base, ua.test, ub.base, ub.test sets run command:

```
python benchmark/evaluate.py
```