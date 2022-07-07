# RMHN
This repository contains the source code of our paper, and we also gives a link of the datasets.



## Datasets

> Please first download the datasets [here](https://www.aliyundrive.com/s/h93t537jRJ7 ) and extract them into `data/` directory.

Initial datasets DBP15K is from [JAPE](https://github.com/nju-websoft/JAPE) and [BootEA](https://github.com/nju-websoft/BootEA).

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" includes:

* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* ref_ent_ids: entity links encoded by ids;
* triples_1: relation triples encoded by ids in source KG (DBP_ZH);
* triples_1_s: remaining relation triples encoded by ids in source KG (S-DBP_ZH);
* triples_2: relation triples encoded by ids in target KG (DBP_EN);
* triples_2_s: remaining relation triples encoded by ids in target KG (S-DBP_EN);
* vectorList.json: the input entity feature matrix initialized by word vectors;

## Environment

* Python>=3.7
* Tensorflow>=2.1.0
* Scipy>=1.4.1
* Numpy>=1.21.5
* json>=0.8.4
* pickle
* pandas>=1.2.4

## Running

For example, to run RMHN on DBP15K (ZH-EN), use the following script:

```
python main.py --dataset DBP15k --lang zh_en
```


> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

## Citation

If you use this model or code, please cite it as follows: