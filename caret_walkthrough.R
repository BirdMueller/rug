#include necessary libraries
library(caret)
library(mlbench)
library(titanic)


set.seed(613)

vars<-7

regr_fmla<-function(maxDegree){
  degVec<-sapply(0:maxDegree,function(power){return(paste0('x',power))})
  return(as.formula(paste0('y~',paste0(degVec,collapse='+'))))
}

#our number of points
N<-1000
#spread x coords out evenly in (0,2*Pi]
xRaw<-(1:N)*2*pi/(N)
xNoise<-0
#compute y = sin(x) and "measurement errors"
yRaw<-sin(xRaw)
yNoise<-0.25*rnorm(N)
#compute final x and y coordinates
xCoord<-xRaw+xNoise
yCoord<-yRaw+yNoise

plot(
  xCoord,
  yCoord,
  xlab = "x",
  ylab = "y",
  main = "y = sin(x)",
  col='blue'
)
lines(xRaw,yRaw)

higherPowers<-data.frame(lapply(0:100,function(power){
  return(sapply(xCoord,function(x){
    return(x**power)
  }))
}))
names(higherPowers)<-sapply(0:100,function(power){return(paste0('x',power))})
higherPowers$y<-yCoord

#plot training error with increasing degree 1 to 100
plot(
  1:100,
  sapply(1:100,function(i){
    regrModel<-lm(regr_fmla(i),higherPowers)
    return((1/N)*sum(abs( yCoord-predict(regrModel,higherPowers) )))
  }),
  xlab = "Polynomial Degree (Number of Independent Variables)",
  ylab = "Sum of Absolute Values of Errors",
  main = "Training Error",
  col='blue'
)

#plot training error with increasing degree 80 to 100
plot(
  80:100,
  sapply(80:100,function(i){
    regrModel<-lm(regr_fmla(i),higherPowers)
    return((1/N)*sum(abs( yCoord-predict(regrModel,higherPowers) )))
  }),
  xlab = "Polynomial Degree (Number of Independent Variables)",
  ylab = "Sum of Absolute Values of Errors",
  main = "Training Error",
  col='blue'
)

#plot 'noiseless error' 1 to 100
plot(
  1:100,
  sapply(1:100,function(i){
    regrModel<-lm(regr_fmla(i),higherPowers)
    return((1/N)*sum(abs( yRaw-predict(regrModel,higherPowers) )))
  }),
  xlab = "Polynomial Degree (Number of Independent Variables)",
  ylab = "Sum of Absolute Values of Errors",
  main = "'Noiseless' Error",
  col='blue'
)

#plot 'noiseless error' 80 to 100
plot(
  1:10,
  sapply(1:10,function(i){
    regrModel<-lm(regr_fmla(i),higherPowers)
    return((1/N)*sum(abs( yRaw-predict(regrModel,higherPowers) )))
  }),
  xlab = "Polynomial Degree (Number of Independent Variables)",
  ylab = "Sum of Absolute Values of Errors",
  main = "'Noiseless' Error",
  col='blue'
)

#plot 'noiseless error' 80 to 100
plot(
  80:100,
  sapply(80:100,function(i){
    regrModel<-lm(regr_fmla(i),higherPowers)
    return((1/N)*sum(abs( yRaw-predict(regrModel,higherPowers) )))
  }),
  xlab = "Polynomial Degree (Number of Independent Variables)",
  ylab = "Sum of Absolute Values of Errors",
  main = "'Noiseless' Error",
  col='blue'
)

###
#now proceed with our data split into train and test sets
###


#take the first three quarters of my N points
inTrain<-1:floor(0.75*N)
#create data frames for training and then testing afterwards
training<-higherPowers[inTrain,]
testing<-higherPowers[-inTrain,]

#testing error 3 to 30
plot(
  3:50,
  sapply(3:50,function(i){
    regrModel<-lm(regr_fmla(i),training)
    return((1/N)*sum(abs( yCoord[-inTrain]-predict(regrModel,testing) )))
  }),
  xlab = "Polynomial Degree (Number of Independent Variables)",
  ylab = "Sum of Absolute Values of Errors",
  main = "Testing Error",
  col='blue'
)

#testing error 3 to 7
plot(
  3:vars,
  sapply(3:vars,function(i){
    regrModel<-lm(regr_fmla(i),training)
    return((1/N)*sum(abs( yCoord[-inTrain]-predict(regrModel,testing) )))
  }),
  xlab = "Polynomial Degree (Number of Independent Variables)",
  ylab = "Sum of Absolute Values of Errors",
  main = "Testing Error",
  col='blue'
)

#residues versus data with noise
plot(
  xCoord,
  yCoord-predict(lm(regr_fmla(vars),training),higherPowers),
  xlab = "x",
  ylab = "y",
  main = "Residues, Degree 5 vs Degree 7",
  col='red'
)
points(
  xCoord,
  yCoord-predict(lm(regr_fmla(5),training),higherPowers),
  col='green'
)
abline(h=0)


###
#now using caret
###

#make sure dependent variable is a factor 
#make sure it references famous people
titanic_train$outFactor<-as.factor(
  sapply(titanic_train$Survived,function(row){
    if(row){
      return('Winslet')
    }else{
      return('DiCaprio')
    }
}))

#find a random three quarters subset of all indices
inTrainTitanic<-createDataPartition(
  y=titanic_train$outFactor,
  p=0.75,
  list=FALSE
)
#create data frames of testing and training data
trainTitanic<-titanic_train[inTrainTitanic,]
testTitanic<-titanic_train[-inTrainTitanic,]

rf <- train(
  outFactor ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = trainTitanic,
  method = "rf"
)
confusionMatrix(
  data = predict(rf,newdata = testTitanic), 
  na.omit(testTitanic)$outFactor
)

models<-c('rpart','rf','xgbTree')
modelList<-lapply(models,function(name){
  return(
    train(
      outFactor ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
      data = trainTitanic,
      method = name
    )   
  )
})
modelListConf<-sapply(modelList,function(model){
  return(
    confusionMatrix(
      data = predict(model,newdata = testTitanic), 
      na.omit(testTitanic)$outFactor
    )[['byClass']][1]
  )
})
names(modelListConf)<-models
barplot(
  modelListConf, 
  main="Sensitivity", 
  xlab="Algorithm Code"
)

grid <- expand.grid(
  mtry=c(2,3)
)
ctrl <- trainControl(
  method = "repeatedcv",
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
rfTwo <- train(
  outFactor ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = titanic_train,
  method = "rf",
  trControl = ctrl,
  tuneGrid = grid
)
