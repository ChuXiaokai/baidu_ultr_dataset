# A Large Scale Search Dataset from Baidu Search Engine ([Homepage](www.baidu-ultr.com)).
This repo contains code & dataset accompaning the paper, [A Large Scale Search Dataset for Unbiased Learning to Rank](https://arxiv.org/). 

### Dependencies
This code requires the following:

### Usage
Quick Start: ""

### The Pre-trained Language Model
You can download the pre-trained language model from the table below:

|   |H=768|
|---:|---:|
| **L=12** |[Baidu_ULTR_Base_12_768](https://drive.google.com/file/d/1Ft7DGH66fVOqQCMym57OeApp5NA2Y3Oc/view?usp=sharing)|

### Large Scale Web Search Session Data
The large scale web search session are aviable at [here](https://drive.google.com/drive/folders/1Q3bzSgiGh1D5iunRky6mb89LpxfAO73J?usp=sharing).
The search session is organized as:
```
Qid, Query, Query_reformulation
Pos 1, Url_md5, Title, Abstract, Multimedia_type, Click, -, -, Skip, SERP Height, Displayed Time, Displayed Time Middle,First Click,Displayed Count, SERP's Max Show Height, Slipoff Count After Click, Dwelling Time , Displayed Time Top, SERP to Top , Displayed Count Top, Displayed Count Bottom, Slipoff Count, -, Final Click, Displayed Time Bottom, Click Count, Displayed Count, -, Last Click , Reverse Display Count, Displayed Count Middle, -
Pos 2, Url_md5, Title, Abstract, Multimedia_type, Click, -, -, Skip, SERP Height, Displayed Time, Displayed Time Middle,First Click,Displayed Count, SERP's Max Show Height, Slipoff Count After Click, Dwelling Time , Displayed Time Top, SERP to Top , Displayed Count Top, Displayed Count Bottom, Slipoff Count, -, Final Click, Displayed Time Bottom, Click Count, Displayed Count, -, Last Click , Reverse Display Count, Displayed Count Middle, -
......
Pos N, Url_md5, Title, Abstract, Multimedia Type, Click, -, -, Skip, SERP Height, Displayed Time, Displayed Time Middle,First Click,Displayed Count, SERP's Max Show Height, Slipoff Count After Click, Dwelling Time , Displayed Time Top, SERP to Top , Displayed Count Top, Displayed Count Bottom, Slipoff Count, -, Final Click, Displayed Time Bottom, Click Count, Displayed Count, -, Last Click , Reverse Display Count, Displayed Count Middle, -
```
|Column Id|Explaination|Remark|
|:---|:---|:---|
|Qid|query id||
|Query|the user issued query|Sequential token ids seprated by "\x01".|
|Query_reformulation|The subsequent queries issued by users under the same search goal. A session can have multiple queries.|Sequential token ids seprated by "\x01".|
|Pos|The document’s displaying order on the screen.|\[1,30\]|
|Url_md5|The md5 for identifying the url||
|Title|The title of document.|Sequential token ids seprated by "\x01".|
|Abstract|A query-related brief introduction of document under the title.|Sequential token ids seprated by "\x01".|
|Multimedia Type|The type of url, for example, advertisement, videos, maps.|int|
|Click|Whether user clicked the document.|\[0,1\]|
|-|-|-|
|-|-|-|
|Skip|Whether user skipped the document on the screen.|\[0,1\]|
|SERP Height|The vertical pixels of SERP on the screen.|Continuous Value|
|Displayed Time|The document's display time on the screen.|Continuous Value|
|Displayed Time Middle|The document’s display time on the middle 1/3 of screen.|Continuous Value|
|First Click|The identifier of users’ first click in a query.|\[0,1\]|
|Displayed Count|The document’s display count on the screen.|Discrete Number|
|SERP's Max Show Height|The max vertical pixels of SERP on the screen.|Continuous Value|
|Slipoff Count After Click |The count of slipoff after user click the document.|Discrete Number|
|Dwelling Time|The length of time a user spends looking at a document after they’ve clicked a link on a SERP page, but before clicking back to the SERP results.|Continuous Value|
|Displayed Time Top|The document’s display time on the top 1/3 of screen.|Continuous Value|
|SERP to Top|The vertical pixels of the SERP to the top of the screen.|Continuous Value|
|Displayed Count Top|The document’s display count on the top 1/3 of screen.|Discrete Number|
|Displayed Count Bottom|The document’s display count on the bottom 1/3 of screen.|Discrete Number|
|Slipoff Count|The count of document being sliped off the screen.||
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

### Expert Annotation Dataset for Validation, Test or Model Fine-tuning.
The expert annotation dataset is aviable at [here](https://drive.google.com/drive/folders/1AmLTDNVltS02cBMIVJJLfVc_xIrLA2cL?usp=sharing).
The Schema of the [nips_annotation_data_0522.txt](https://drive.google.com/file/d/1hdWRRSMrCnQxilYfjTx8RhW3XTgiSd9Q/view?usp=sharing):
|Column Id|Explaination|Remark|
|---:|---:|---:|
|1|query id|Explaination||
|2|query tokens|sep by "\x01"|
|3|abstract tokens|sep by "\x01"|
|4|annotation label|\[0,4\]|
|5|query bucket|\[0,9\]|
The [unigram_dict_0510_tokens.txt](https://drive.google.com/file/d/1HZ7l7UDMH9WvLVoDu-_uqLNjF5gtBe2g/view?usp=sharing) is a unigram
set that records the high-frequency words using the desensitization token id.

### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/ChuXiaokai/baidu_ultr_dataset/issues).
