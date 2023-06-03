# 决策树
library(C50)
credit$default <- factor(credit$default,levels = c(1,2),labels = c("no","yes"))
# 载入数据
credit <- read.csv(file.choose())
credit.rm <- credit[,18:21]
credit <- credit[,1:17]
str(credit)
table(credit$checking_balance)
table(credit$savings_balance)
summary(credit$months_loan_duration)
summary(credit$amount)
table(credit$default)
# 数据准备
set.seed(12345)# 确保与书本数据一致
# 数据打散
credit_rand <- credit[order(runif(1000)),]
# 确认数据无误
summary(credit$amount)
summary(credit_rand$amount)
head(credit$amount)
head(credit_rand$amount)
# 创建训练集和测试集
credit_train <- credit_rand[1:900,]
credit_test <- credit_rand[901:1000,]
prop.table(table(credit_train$default))
prop.table(table(credit_test$default))
credit_train$default<-as.factor(credit_train$default)
credit_model <- C5.0(credit_train[-17],credit_train$default)# 去除类变量
credit_model
summary(credit_model)
# 评估模型性能
credit_pred <- predict(credit_model,credit_test)
library(gmodels)
CrossTable(credit_test$default,credit_pred,prop.chisq = FALSE,prop.r = FALSE,prop.c = FALSE,dnn = c('actual default','predicted default'))
# 提高模型性能
credit_boost10 <- C5.0(credit_train[-17],credit_train$default,trials = 10)
credit_boost10
summary(credit_boost10)
credit_boost_pred10 <- predict(credit_boost10,credit_test)
CrossTable(credit_test$default,credit_boost_pred10,prop.chisq = FALSE,prop.r = FALSE,prop.c = FALSE,dnn = c('actual default','predicted default'))
# 假设贷款违约者使银行损失4倍
error_cost <- matrix(c(0,1,4,0),nrow = 2)
credit_cost <- C5.0(credit_train[-17],credit_train$default,costs = error_cost)
credit_cost_pred <- predict(credit_cost,credit_test)
CrossTable(credit_test$default,credit_cost_pred,prop.chisq = FALSE,prop.r = FALSE,prop.c = FALSE,dnn = c('actual default','predicted default'))
