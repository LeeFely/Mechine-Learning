kNN临近分类

library(class)#加载R包
library(gmodels)

wbcd <- read.csv("breast-cancer-wisconsin.data.csv",
                 stringsAsFactors = FALSE)# 读入数据
str(wbcd)

wbcd <- wbcd[-1]# 去除ID
table(wbcd$Diagnosis)
wbcd$Diagnosis <- factor(wbcd$Diagnosis,levels = c("B","M"),
                         labels = c("Benign","Malignant"))
round(prop.table(table(wbcd$Diagnosis)) * 100,digits = 1)
summary(wbcd[c("radius_mean","area_mean","smoothness_mean")])

normalize <- function(x){# 编写min-max标准化函数
  return ((x - min(x)) / (max(x)-min(x)))
}

wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
summary(wbcd_n$area_mean)

wbcd_train <- wbcd_n[1:469,]# 训练集
wbcd_test <- wbcd_n[470:569,]# 测试集

wbcd_train_labels <- wbcd[1:469,1]
wbcd_test_labels <- wbcd[470:569,1]
wbcd_test_pred <- knn(train = wbcd_train,test = wbcd_test,
                      cl = wbcd_train_labels,k = 21)

CrossTable(x=wbcd_test_labels,y=wbcd_test_pred,prop.chisq = FALSE)
wbcd_z <- as.data.frame(scale(wbcd[-1]))#z-score标准化
summary(wbcd_z$area_mean)
wbcd_train1 <- wbcd_z[1:469,]
wbcd_test1 <- wbcd_z[470:569,]
wbcd_train1_labels <- wbcd[1:469,1]
wbcd_test1_labels <- wbcd[470:569,1]
wbcd_test1_pred <- knn(train = wbcd_train1,
                       test = wbcd_test1,cl = wbcd_train1_labels,k=21)
CrossTable(x=wbcd_test1_labels,y=wbcd_test1_pred,prop.chisq = FALSE)


## 网络代码
近邻分类
简⾔之，就是将未标记的案例归类为与它们最近相似的、带有标记的案例所在的类。
应⽤领域：
1.计算机视觉：包含字符和⾯部识别等
2.推荐系统：推荐受众喜欢电影、美⾷和娱乐等
3.基因⼯程：识别基因数据的模式，⽤于发现特定的蛋⽩质或疾病等
K最近邻(kNN，k-NearestNeighbor)算法
K最近邻分类算法是数据挖掘分类技术中最简单的⽅法之⼀。所谓K最近邻。
kNN算法的核⼼思想是如果⼀个样本在特征空间中的k个最相邻的样本中的⼤多数属于某⼀个类别，则该样本也属于这个类别（类似投票），并具有这个类别上样本的特性。
该⽅法在确定分类决策上只依据最邻近的⼀个或者⼏个样本的类别来决定待分样本所属的类别。
kNN⽅法在类别决策时，只与极少量的相邻样本有关。
由于kNN⽅法主要靠周围有限的邻近的样本，⽽不是靠判别类域的⽅法来确定所属类别的，因此对于类域的交叉或重叠较多的待分样本集来说，kNN⽅法较其他⽅法更为适合。
R的实现
具体的算法原理本⽂就不赘述了，下⾯进⾏⼀个R中knn算法的⼩实验。数据使⽤UCI的[乳腺癌特征数据集](http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)

dir <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
wdbc.data <- read.csv(dir,header = F)
names(wdbc.data) <- c('ID','Diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean',
                      'symmetry_mean','fractal dimension_mean','radius_sd','texture_sd','perimeter_sd','area_sd','smoothness_sd','compactness_sd','concavity_sd','concave points_sd',
                      'symmetry_sd','fractal dimension_sd','radius_max_mean','texture_max_mean','perimeter_max_mean','area_max_mean','smoothness_max_mean',
                      'compactness_max_mean','concavity_max_mean','concave points_max_mean','symmetry_max_mean','fractal dimension_max_mean')
table(wdbc.data$Diagnosis) ## M = malignant, B = benign
# 将⽬标属性编码因⼦类型
wdbc.data$Diagnosis <- factor(wdbc.data$Diagnosis,levels =c('B','M'),labels = c(B = 'benign',M = 'malignant'))
wdbc.data$Diagnosis
table(wdbc.data$Diagnosis)
prop.table(table(wdbc.data$Diagnosis))*100 ## prop.table():计算table各列的占⽐
round(prop.table(table(wdbc.data$Diagnosis))*100,digit =2) ## 保留⼩数点后两位，round()：digit =2
str(wdbc.data)

# min-max标准化:(x-min)/(max-min)
normalize <- function(x) { return ((x-min(x))/(max(x)-min(x))) }
normalize(c(1, 3, 5)) ## 测试函数有效性
wdbc.data.min_max <- as.data.frame(lapply(wdbc.data[3:length(wdbc.data)],normalize))
wdbc.data.min_max$Diagnosis <- wdbc.data$Diagnosis
str(wdbc.data.min_max)

# train
set.seed(3) ## 设⽴随机种⼦
train_id <- sample(1:length(wdbc.data.min_max$area_max_mean), length(wdbc.data.min_max$area_max_mean)*0.7)
train <- wdbc.data.min_max[train_id,] # 70%训练集
summary(train)
train_labels <- train$Diagnosis
train <- wdbc.data.min_max[train_id, - length(wdbc.data.min_max)]
summary(train)
# test
test <- wdbc.data.min_max[-train_id,]
test_labels <- test$Diagnosis
test <- wdbc.data.min_max[-train_id,-length(wdbc.data.min_max)]
summary(test)

library(class)
test_pre_labels <- knn(train,test,train_labels,k=20) ## 数据框，K个近邻投票,欧⽒距离

library(gmodels)
CrossTable(x = test_labels, y = test_pre_labels, prop.chisq = F)
