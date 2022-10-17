# A Large Scale Search Dataset from Baidu Search Engine
This repo contains code & dataset accompaning the paper, [A Large Scale Search Dataset for Unbiased Learning to Rank](https://arxiv.org/pdf/2207.03051.pdf). 

### Dependencies
This code requires the following:
- Python 3.6+
- Pytorch 1.10.2 + CUDA 10.2

### Quick Start

#### 0. Prepare the corpus
Suppose your have downloaded [the Web Search Session Data](https://drive.google.com/drive/folders/1Q3bzSgiGh1D5iunRky6mb89LpxfAO73J?usp=sharing) (training data) and [annotation_data_0522.txt](https://drive.google.com/file/d/1hdWRRSMrCnQxilYfjTx8RhW3XTgiSd9Q/view?usp=sharing) (test data) on Google drive.

Moreover, we provide the resource for those who cannot access google drive. [training data](https://searchscience.baidu.com/dataset_ultr_train.html) [test data](https://searchscience.baidu.com/baidu_ultr/labeled_dataset/test_data.txt) [unigram dict](https://searchscience.baidu.com/baidu_ultr/labeled_dataset/unigram_dict_0510_tokens.txt).


First, move all the zip file into dir './data/train_data/', e.g.,
> ```mv yourpath/*.gz ./data/train_data/```

Second, move the file **part-00000.gz** into './data/click_data/', we will treat it as one of the validation set.
> ```mv ./data/train_data/part-00000.gz ./data/click_data/part-00000.gz``` 

Finally, split the annotated data [annotation_data_0522.txt](https://drive.google.com/file/d/1hdWRRSMrCnQxilYfjTx8RhW3XTgiSd9Q/view?usp=sharing) into test and validation set. Move them into dir './data/annotate_data/'
> ```mv test_data.txt ./data/annotate_data/```
> ```mv val_data.txt ./data/annotate_data/```

#### Pretrain Transformer
We adopt two tasks **CTR** prediction and **MLM** to pretrain the Transformer. For example, to pretrain a 12-Layers Transformer, you can type the following command
> ``` python pretrain.py --emb_dim 768 --nlayer 12 --nhead 12 --dropout 0.1 ```
#### Training Baselines
We select five representative baselines to test this dataset, which are Naive, IPW, DLA, REM and PairD. The code and hyper-parameters setting refer to [ULTR-Community](https://github.com/ULTR-Community/ULTRA_pytorch). For example, to train a base model IPW, you can type the following command
> ``` python finetune.py --method_name IPW```

The explanation of the input parameters, you can refer to args.py.

### The Pre-trained Language Model
You can download the pre-trained language model from the table below:

|   |Head=12|
|---:|---:|
| **Layer=12** |[Baidu_ULTR_Base_12L_12H_768Emb](https://drive.google.com/file/d/1KWOd2TsFwgWIAMDBv9mVXQ4D5h6s5tWy/view?usp=sharing)|
| **Layer=6** |[Baidu_ULTR_Base_6L_12H_768Emb](https://drive.google.com/file/d/1MBGfCDH7giODsnfThtotIfpM6KI_szFw/view?usp=sharing)|
| **Layer=3** |[Baidu_ULTR_Base_3L_12H_768Emb](https://drive.google.com/file/d/1f9F3eN0V20iCtoxNMkrPDSSZss5nno8C/view?usp=sharing)|

### Train Data --- Large Scale Web Search Session Data
The large scale web search session are available at [here](https://drive.google.com/drive/folders/1Q3bzSgiGh1D5iunRky6mb89LpxfAO73J?usp=sharing).
The search session is organized as:
```
Qid, Query, Query Reformulation
Pos 1, URL MD5, Title, Abstract, Multimedia Type, Click, -, -, Skip, SERP Height, Displayed Time, Displayed Time Middle, First Click, Displayed Count, SERP's Max Show Height, Slipoff Count After Click, Dwelling Time , Displayed Time Top, SERP to Top , Displayed Count Top, Displayed Count Bottom, Slipoff Count, -, Final Click, Displayed Time Bottom, Click Count, Displayed Count, -, Last Click , Reverse Display Count, Displayed Count Middle, -
Pos 2, URL MD5, Title, Abstract, Multimedia Type, Click, -, -, Skip, SERP Height, Displayed Time, Displayed Time Middle, First Click, Displayed Count, SERP's Max Show Height, Slipoff Count After Click, Dwelling Time , Displayed Time Top, SERP to Top , Displayed Count Top, Displayed Count Bottom, Slipoff Count, -, Final Click, Displayed Time Bottom, Click Count, Displayed Count, -, Last Click , Reverse Display Count, Displayed Count Middle, -
......
Pos N, URL MD5, Title, Abstract, Multimedia Type, Click, -, -, Skip, SERP Height, Displayed Time, Displayed Time Middle, First Click, Displayed Count, SERP's Max Show Height, Slipoff Count After Click, Dwelling Time , Displayed Time Top, SERP to Top , Displayed Count Top, Displayed Count Bottom, Slipoff Count, -, Final Click, Displayed Time Bottom, Click Count, Displayed Count, -, Last Click , Reverse Display Count, Displayed Count Middle, -
# SERP is the abbreviation of search result page.
```


|Column Id|Explaination|Remark|
|:---|:---|:---|
|Qid|query id||
|Query|The user issued query|Sequential token ids separated by "\x01".|
|Query Reformulation|The subsequent queries issued by users under the same search goal.|Sequential token ids separated by "\x01".|
|Pos|The document’s displaying order on the screen.|\[1,30\]|
|Url_md5|The md5 for identifying the url||
|Title|The title of document.|Sequential token ids separated by "\x01".|
|Abstract|A query-related brief introduction of the document under the title.|Sequential token ids separated by "\x01".|
|Multimedia Type|The type of url, for example, advertisement, videos, maps.|int|
|Click|Whether the user clicked the document.|\[0,1\]|
|-|-|-|
|-|-|-|
|Skip|Whether the user skipped the document on the screen.|\[0,1\]|
|SERP Height|The vertical pixels of SERP on the screen.|Continuous Value|
|Displayed Time|The document's display time on the screen.|Continuous Value|
|Displayed Time Middle|The document’s display time on the middle 1/3 of the screen.|Continuous Value|
|First Click|The identifier of users’ first click in a query.|\[0,1\]|
|Displayed Count|The document’s display count on the screen.|Discrete Number|
|SERP's Max Show Height|The max vertical pixels of SERP on the screen.|Continuous Value|
|Slipoff Count After Click |The count of slipoff after user click the document.|Discrete Number|
|Dwelling Time|The length of time a user spends looking at a document after they’ve clicked a link on a SERP page, but before clicking back to the SERP results.|Continuous Value|
|Displayed Time Top|The document’s display time on the top 1/3 of screen.|Continuous Value|
|SERP to Top|The vertical pixels of the SERP to the top of the screen.|Continuous Value|
|Displayed Count Top|The document’s display count on the top 1/3 of screen.|Discrete Number|
|Displayed Count Bottom|The document’s display count on the bottom 1/3 of screen.|Discrete Number|
|Slipoff Count|The count of document being slipped off the screen.||
|-|-|-|
|Final Click |The identifier of users’ last click in a query session.||
|Displayed Time Bottom|The document’s display time on the bottom 1/3 of screen.|Continuous Value|
|Click Count|The document’s click count.|Discrete Number|
|Displayed Count|The document’s display count on the screen.|Discrete Number|
|-|-|-|
|Last Click |The identifier of users’ last click in a query.|Discrete Number|
|Reverse Display Count|The document’s display count of user view with a reverse browse order from bottom to the top.|Discrete Number|
|Displayed Count Middle|The document’s display count on the middle 1/3 of screen.|Discrete Number|
|-|-|-|

### Test Data --- Expert Annotation Dataset for Validation, 
The expert annotation dataset is aviable at [here](https://drive.google.com/drive/folders/1AmLTDNVltS02cBMIVJJLfVc_xIrLA2cL?usp=sharing).
The Schema of the [annotation_data_0522.txt](https://drive.google.com/file/d/1hdWRRSMrCnQxilYfjTx8RhW3XTgiSd9Q/view?usp=sharing):
|Columns|Explaination|Remark|
|:---|:---|:---|
|Qid|The uniq id for every query.| An uniq id.|
|Query|The user issued query|Sequential token ids separated by "\x01".|
|Title|The title of document.|Sequential token ids separated by "\x01".|
|Abstract|A query-related brief introduction of the document under the title.|Sequential token ids separated by "\x01".|
|Label|Expert annotation label.|\[0,4\]|
|Bucket|The queries are descendingly split into 10 buckets according to their monthly search frequency, i.e., bucket 0, bucket 1, and bucket 2 are high-frequency queries while bucket 7, bucket 8, and bucket 9 are the tail queries|\[0,9\]|

The [unigram_dict_0510_tokens.txt](https://drive.google.com/file/d/1HZ7l7UDMH9WvLVoDu-_uqLNjF5gtBe2g/view?usp=sharing) is a unigram set that records the high-frequency words using the desensitization token id.

### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/ChuXiaokai/baidu_ultr_dataset/issues).
