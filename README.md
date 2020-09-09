# Title_Generator_CN
分别使用TextRank、BiLSTM和UniLM实现中文文章标题自动生成

## 说明：
* TG_BiLSTM: seq2seq模型标题生成，encoder与decoder主要使用了双层双向LSTM

  Corpus: [新闻语料json版 (news2016zh)](https://github.com/brightmart/nlp_chinese_corpus)
* TG_TextRank: 使用基本的TextRank模型抽取文章中的关键句作为标题
* TG_TextRank+W2V: TextRank算法在计算句子之间的相似度作为边的权重时，将词共现频率改为使用Word2Vec计算相似度
* TG_UniLM: 使用[UniLM语言模型](https://arxiv.org/abs/1905.03197)进行标题生成

  Corpus: [THUCNews数据集](http://thuctc.thunlp.org/)

## 测试：
* 测试数据集个人构造3344条新闻及标题
* 使用[bert-as-service](https://github.com/hanxiao/bert-as-service)对原标题和生成的标题进行编码，然后计算平均余弦相似度
* 测试结果：
  ```
  sim_bilstm: 93.53%
  sim_textrank: 93.67%
  sim_trw2v: 93.93%
  sim_unilm: 95.65%
  ```