# A Large Scale Search Dataset from Baidu Search Engine ([Homepage](www.baidu-ultr.com)).
This repo contains code & dataset accompaning the paper, [A Large Scale Search Dataset for Unbiased Learning to Rank](https://arxiv.org/). 

### Dependencies
This code requires the following:

### Usage
Quick Start: ""


### Large Scale Web Search Session Data
The large scale web search session are aviable at [here](https://drive.google.com/drive/folders/1Q3bzSgiGh1D5iunRky6mb89LpxfAO73J?usp=sharing).
The search session is organized as:
```
qid, query, query_reformulation
pos 1, url_md5, title, abstract, page_type, click, fm, g_st, skip, height, view_time, view_time_middle, count_first_click,view_count, max_show_height, slipoff_after_click, click_time, view_time_up, top, view_count_up, view_count_down, slipoff_times, srcid, final_click_count, view_time_down, click_count, full_view_count, tpl, count_last_click, reverse_view_cnt, view_count_middle, full_view_time
pos 2, url_md5, title, abstract, page_type, click, fm, g_st, skip, height, view_time, view_time_middle, count_first_click,view_count, max_show_height, slipoff_after_click, click_time, view_time_up, top, view_count_up, view_count_down, slipoff_times, srcid, final_click_count, view_time_down, click_count, full_view_count, tpl, count_last_click, reverse_view_cnt, view_count_middle, full_view_time
......
pos N
```
|Column Id|Explaination|Remark|
|---:|---:|---:|
|qid|query id||
|query|query tokens|sep by "\x01"|
|query_reformulation|query reformulation tokens|sep by "\x01"|
|pos N|the ranking position of current url|\[1,30\]|
|url_md5|the md5 for identifying the url||
|title|title tokens|sep by "\x01"|
|abstract|abstract tokens|sep by "\x01"|
|page_type| the multi-media type of the page||
|click|||


### The Pre-trained Language Model
You can download the pre-trained language model from the table below:

|   |H=768|
|---:|---:|
| **L=12** |[**12/768 (Baidu_ULTR_Base)**][12_768]|

### Expert Annotation Dataset 
The expert annotation dataset is aviable at [here]().
The Schema of the [nips_test_0522.txt]:
|Column Id|Explaination|Remark|
|---:|---:|---:|
|1|query id|Explaination||
|2|query tokens|sep by "\x01"|
|3|abstract tokens|sep by "\x01"|
|4|annotation label|\[0,4\]|

### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/ChuXiaokai/baidu_ultr_dataset/issues).
