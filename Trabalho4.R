########################################################################
# INF-0615 - Tarefa 4 - Predição de números                            #
# Alunos: Rafael Fernando Ribeiro                                      #
#         Thiago Gomes Marcal Pereira                                  #
########################################################################
#install.packages("e1071")
#install.packages("neuralnet")
#install.packages("caret")
library(e1071)
library(neuralnet)
library(caret)

# Predict data using model and evaluate
predictAndEvaluateSVM <- function(model, data){
  prediction = predict(model, data)
  prediction = as.numeric(prediction >= 0.5)
  prediction[prediction==0] = "0"
  prediction[prediction==1] = "1"
  
  CM = as.matrix(table(Actual = data$V1, Predicted = prediction))
  
  if (dim(CM)[2] == 1) {
    CM <- cbind(CM, c(0,0))
  }
  
  TPR = CM[2,2] / (CM[2,2] + CM[2,1])
  TNR = CM[1,1] / (CM[1,1] + CM[1,2])
  ACCNorm = mean(c(TPR, TNR))
  
  return(list(CM=CM, ACCNorm=ACCNorm))
}

# Predict data using model and evaluate
predictAndEvaluateNN <- function(model, data){
  nnCompute = compute(model, data[,-1])
  prediction = nnCompute$net.result
  
  prediction[prediction < 0.5] = -1
  prediction[prediction >= 0.5] = 1
  
  CM = as.matrix(table(Actual = data$V1, Predicted = prediction))
  if (dim(CM)[2] == 1) {
    CM <- cbind(CM, c(0,0))
  }
  TPR = CM[2,2] / (CM[2,2] + CM[2,1])
  TNR = CM[1,1] / (CM[1,1] + CM[1,2])
  ACCNorm = mean(c(TPR, TNR))
  
  return(list(CM=CM, ACCNorm=ACCNorm))
}

train.svm_linear <- function(train, val, cost) {
  
  gridSearchSVMModelsL <- list()
  for (c in cost) {
    set.seed(42)
    print(paste("Training ", digit, " vs All - SVM (Liner) - c=", c, sep = ""))
    
    svmModel <- svm(formula = V1 ~ ., data = train, kernel= "linear", cost = c, scale = FALSE)
    accTrain <- predictAndEvaluateSVM(svmModel, train) # 0.50 :(
    accVal <- predictAndEvaluateSVM(svmModel, val)   # 0.50 :(
    print(accTrain)
    print(accVal)
    
    gridSearchSVMModelsL[[toString(c)]] <- svmModel
  }
  return (gridSearchSVMModelsL)
}

train.svm_rbf <- function(train, val, cost, gamma) {
  
  gridSearchSVMModelsR <- list()
  for (c in cost) {
    for (g in gamma) {
      set.seed(42)
      print(paste("Training ", digit, " vs All - SVM (RBF) - c=", c, "-g=", g, sep = ""))
      
      svmModel <- svm(formula = V1 ~ ., data = train, kernel= "radial", cost = c, gamma = g, scale = FALSE)
      accTrain <- predictAndEvaluateSVM(svmModel, train) # 0.50 :(
      accVal <- predictAndEvaluateSVM(svmModel, val)   # 0.50 :(
      print(accTrain)
      print(accVal)
      
      gridSearchSVMModelsR[[paste(toString(c), toString(g))]] <- svmModel
    }
  }
  return (gridSearchSVMModelsR)
}

train.neural_net <- function(train, val, nets) {
  # train NN model
  gridSearchNNModels <- list()
  # (DOn't believe on this result . It must have something wrong :) )
  #nets <- list(c(5,3), c(7,3), c(7,5,3))
  for (net in nets) {
    set.seed(42)
    print(paste("Training ", digit, " vs All - NN - ", toString(net), sep = ""))
    
    nnModel = neuralnet(formula=f, data=train, hidden=net, linear.output=FALSE) 
    #linear.output = FALSE --> classification (apply 'logistic' activation as default)
    accTrain <- predictAndEvaluateNN(nnModel, train) #0.9792
    accVal <- predictAndEvaluateNN(nnModel, val)   #0.9953  
    print(accTrain)
    print(accVal)
    
    gridSearchNNModels[[toString(net)]] <- nnModel
  }
  return (gridSearchNNModels)
}

bestmodel.predict <- function(bestModels, data) {
  pred_value <-  data.frame()
  for (i in DIGITS) {
    set.seed(42)
    nnComputeV <-  compute(bestModels[[toString(i)]], data)
    
    if (nrow(pred_value) == 0) {
      pred_value <- nnComputeV$net.result
    } 
    else {
      pred_value <- cbind(pred_value, nnComputeV$net.result)
    }
  }
  # name the colummns to get class easily on the vooting
  colnames(pred_value) <- 0:9
  
  # vooting method
  # get the column of maximum value on the prediction to get the class 
  return (colnames(pred_value)[apply(pred_value,1,which.max)])
} 

########################################################################
# Métodos auxiliares para visualização das Imagens                     #
########################################################################

# Função para plotar uma imagem a partir de um único feature vector
# imgArray = fv da imagem, com classe na V1 e os pixels em V2 ~ V785
# os pixels podem estar no intervalo [0.0,1.0] ou [0, 255]
plotImage = function(imgArray){
  # Transforma array em uma matrix 28x28 (ignorando a classe em V1)
  imgMatrix = matrix((imgArray[2:ncol(imgArray)]), nrow=28, ncol=28)
  
  # Transforma cada elemento em numeric
  im_numbers <- apply(imgMatrix, 2, as.numeric)
  
  # Girando a imagem apenas p/ o plot
  flippedImg = im_numbers[,28:1]
  
  image(1:28, 1:28, flippedImg, col=gray((0:255)/255), xlab="", ylab="")
  title(imgArray[1])
}

set.seed(42)
#setwd("/Users/thiagom/Documents/Studies/Unicamp/MDC/INF-615/Tarefas/INF0615_Tarefa4/")
setwd("C:\\Users\\rafaelr\\Documents\\INF015\\Tarefa4\\INF0615_Tarefa4")

# Exemplo de leitura do dataset
# A primeira feature/coluna (V1) é a classe 
# As colunas V2 ~ V785 são os pixels de cada imagem 28x28 em tons de cinza. 
data = read.csv("mnist_trainVal.csv", header=FALSE)

#Inspecionando o número de exemplos por classe
summary(as.factor(data[,1]))

#Ex de uso pegando o primeiro sample
plotImage(data[1,])

##################################
# Para gerar a fórmula concatenando os nomes
#de cada uma das colunas
##################################
# Seleciona todos os nomes
feats <- names(data)

# Concatena o nome de cada feature, ignorando a primeira
f <- paste(feats[2:length(feats)],collapse=' + ')
f <- paste('V1 ~',f)

# Converte para fórmula
f <- as.formula(f)
f

########################################################################
# Desenvolvimento dos Modelos                                          #
########################################################################

empty <- TRUE
# split the data from each class into train and val
for (i in 0:9) {
  numberData <- data[data[,1]==i, ]
  set.seed(42)
  idx <- sample(1:nrow(numberData), 0.8*nrow(numberData))
  print (length(idx))
  trainNbrData <- numberData[idx,]
  valNbrData <- numberData[-idx,]
  if (empty) {
    trainData <- trainNbrData
    valData <- valNbrData
    empty <- FALSE
  } else {
    trainData <- rbind(trainData, trainNbrData)
    valData <- rbind(valData, valNbrData)
  }
}

# normalize data dividing by 255 as all images are from 0~255
trainDataNorm <- trainData[,-1] / 255.0
valDataNorm <- valData[,-1] / 255.0


#digit <- 1 # for tests purpose
nnModels <- list()
svmModelsL <- list()
svmModelsR <- list()
accPerDigit <- data.frame(digit=numeric(10), accTrain=numeric(10) , accVal=numeric(10))
DIGITS <- 0:9
for (digit in DIGITS) {

  # get train and val data for positive class data
  trainDataNormDigit = trainDataNorm[trainData$V1 == digit,]
  valDataNormDigit = valDataNorm[valData$V1 == digit,]
  
  empty <- TRUE
  # get train and val data for negative class data
  for (i in DIGITS[DIGITS != digit]) {
    set.seed(42)
    trainDataNormNonDigitAux = trainDataNorm[trainData$V1 == i,]
    valDataNormNonDigitAux = valDataNorm[valData$V1 == i,]
    idxTrain <- sample(1:nrow(trainDataNormNonDigitAux), 666) # the number of the beast!!!
    idxVal <- sample(1:nrow(valDataNormNonDigitAux), 200) 
    
    if (empty) {
      trainDataNormNonDigit <- trainDataNormNonDigitAux[idxTrain,]
      valDataNormNonDigit <- valDataNormNonDigitAux[idxVal,]
      empty <- FALSE
    }
    else {
      trainDataNormNonDigit <- rbind(trainDataNormNonDigit, trainDataNormNonDigitAux[idxTrain,])
      valDataNormNonDigit <- rbind(valDataNormNonDigit, valDataNormNonDigitAux[idxVal,])
    }
  }
  
  # bind all rows of training data
  pos <- cbind(V1=rep(1, nrow(trainDataNormDigit)), trainDataNormDigit)
  neg <- cbind(V1=rep(-1, nrow(trainDataNormNonDigit)), trainDataNormNonDigit)
  trainDataFinal <- rbind(pos, neg)
  
  # bind all rows of validation data
  pos <- cbind(V1=rep(1, nrow(valDataNormDigit)), valDataNormDigit)
  neg <- cbind(V1=rep(-1, nrow(valDataNormNonDigit)), valDataNormNonDigit)
  valDataFinal <- rbind(pos, neg)

  # train SVM model (need to grid search for better parameters)
  # Check to do PCA
  # check to extract feature like HOG
  
  # grid search
  #svmModelsL[[toString(digit)]] <- train.svm_linear(trainDataFinal, valDataFinal, c(0.0001, 0.001, 0.01, 0.1, 1))
  # train best model
  #svmModelsL[[toString(digit)]] <- train.svm_linear(trainDataFinal, valDataFinal, c(0.001))
  
  # grid search
  #svmModelsR[[toString(digit)]] <- train.svm_rbf(trainDataFinal, valDataFinal, c(0.0001, 0.001, 0.01, 0.1, 1), c(0.0001, 0.001, 0.01, 0.1, 1))
  # train best model
  #svmModelsR[[toString(digit)]] <- train.svm_rbf(trainDataFinal, valDataFinal, c(0.001), c( 0.01))
  
  # grid search
  #nnModels[[toString(digit)]] <- train.neural_net(trainDataFinal, valDataFinal, list(c(5,3), c(7,3), c(7,5,3)))
  # train best model
  #nnModels[[toString(digit)]] <- train.neural_net(trainDataFinal, valDataFinal, list(c(5,3)))

  # check on confusion matrix if some number is more difficult to predict 
  # and train a special model for it
}

# need to choose best models
bestModels <- list()
for (i in DIGITS) {
  bestModels[[toString(i)]] <- nnModels[[toString(i)]][["5, 3"]]
}

# predict class for each model on the training and validation data (will create a function for it) 
train_pred <-  bestmodel.predict(bestModels, trainDataNorm)
val_pred <-  bestmodel.predict(bestModels, valDataNorm)

train_pred <- as.factor(train_pred)
val_pred <- as.factor(val_pred)

train_true <- as.factor(trainData$V1)
val_true <- as.factor(valData$V1)

# generate confusion matrix
# print confusion matrix
# final model prediction and confusion matrix on Training data
table(train_true, train_pred, dnn=c("True Values", "Prediction"))
# final model prediction and confusion matrix on Validation data
table(val_true, val_pred, dnn=c("True Values", "Prediction"))

confusionMatrix(train_true, train_pred)
confusionMatrix(val_true, val_pred)

# need to calculate the normalized accuracy of the train and val data
# print normalized accuracy



# ideia to pass 3 times to model and after majority voot to define the class


  



# load test data
testData = read.csv("mnist_test.csv", header=FALSE)
# Normalize test data (/255.0)
testDataNorm <- testData[,-1] / 255.0
# final model prediction 
test_pred <-  bestmodel.predict(bestModels, testDataNorm)
test_pred <- as.factor(test_pred)
test_true <- as.factor(testData$V1)
# confusion matrix on Test data
table(test_true, test_pred, dnn=c("True Values", "Prediction"))
confusionMatrix(test_true, test_pred)