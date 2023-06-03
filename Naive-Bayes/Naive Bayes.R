# 朴素贝叶斯
sms <- read.csv("spam.csv",stringsAsFactors = FALSE)
str(sms)
sms$type <- factor(sms$type)
sms$text<-iconv(x = sms$text,from = "WINDOWS-1252",to = "UTF-8")
str(sms$type)
table(sms$type)
library(tm)
sms_corpus <- Corpus(VectorSource(sms$text))# 创建语料库
# Corpus()创建R对象储存文本文档
# VectorSource()指示Corpus使用向量sms$text信息
print(sms_corpus)
inspect(sms_corpus)#查看语料库
corpus_clean <- tm_map(sms_corpus,tolower)# 全部小写
corpus_clean <- tm_map(sms_corpus,removeNumbers)# 去除全部数字
corpus_clean <- tm_map(sms_corpus,removeWords,stopwords())# 去除停用词
corpus_clean <- tm_map(sms_corpus,removePunctuation)# 去除标点符号
corpus_clean <- tm_map(sms_corpus,stripWhitespace)# 去除大于一个的空格，只留一个
sms_dlt <- DocumentTermMatrix(corpus_clean)# 创建稀疏矩阵

data<-read.csv("spam.csv",stringsAsFactors = FALSE)
str(data)
data

data$type<-factor(data$type)
data$text<-iconv(x = data$text,from = "WINDOWS-1252",to = "UTF-8")

library(tm)
data_corpus_text<-Corpus(VectorSource(data$text))

data_corpus_text_train<-data_corpus_text[1:5000]
data_corpus_text_test<-data_corpus_text[5001:5558]

dtm_train<-DocumentTermMatrix(x = data_corpus_text_train)
dtm_test<-DocumentTermMatrix(x = data_corpus_text_test)

inspect(dtm_test)

conver_to_matrix<-function(x){
  x<-ifelse(x>0,1,0)
  x<-factor(x,levels = c(0,1),labels = c("\"No\"","\"Yes\""))
  return(x)
}

m_dtm_train<-apply(X = dtm_train,MARGIN = 2,FUN = conver_to_matrix)

m_dtm_test<-apply(X = dtm_test,MARGIN = 2,FUN = conver_to_matrix)


library(e1071)

classifier<-naiveBayes(x = m_dtm_train,y = data$type[1:5000])
predictions<-predict(object = classifier,newdata = m_dtm_test)

library(gmodels)

CrossTable(x = data$type[5001:5558],y = predictions,dnn = c("Actual","Predict"))