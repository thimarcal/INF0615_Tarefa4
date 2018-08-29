########################################################################
# INF-0615 - Tarefa 4 - Predição de números                            #
# Alunos: Rafael Fernando Ribeiro                                      #
#         Thiago Gomes Marcal Pereira                                  #
########################################################################
library(e1071)
library(neuralnet)

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

set.seed(42)
#setwd("/Users/thiagom/Documents/Studies/Unicamp/MDC/INF-615/Tarefas/INF0615_Tarefa4/")
setwd("C:\\Users\\rafaelr\\Documents\\INF015\\Tarefa4\\INF0615_Tarefa4")

# Exemplo de leitura do dataset
# A primeira feature/coluna (V1) é a classe 
# As colunas V2 ~ V785 são os pixels de cada imagem 28x28 em tons de cinza. 
data = read.csv("mnist_trainVal.csv", header=FALSE)

#Inspecionando o número de exemplos por classe
summary(as.factor(data[,1]))

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

DIGITS <- 0:9
digit <- 0

#for (digit in DIGITS) {

  # get train and val data for positive class data
  trainDataNormDigit = trainDataNorm[trainData$V1 == digit,]
  valDataNormDigit = valDataNorm[valData$V1 == digit,]
  
  empty <- TRUE
  # get train and val data for negative class data
  for (i in DIGITS[-(digit+1)]) {
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
  pos <- cbind(V1=rep(1, dim(trainDataNormDigit)[1]), trainDataNormDigit)
  neg <- cbind(V1=rep(-1, dim(trainDataNormNonDigit)[1]), trainDataNormNonDigit)
  trainDataFinal <- rbind(pos, neg)
  
  # bind all rows of validation data
  pos <- cbind(V1=rep(1, dim(valDataNormDigit)[1]), valDataNormDigit)
  neg <- cbind(V1=rep(-1, dim(valDataNormNonDigit)[1]), valDataNormNonDigit)
  valDataFinal <- rbind(pos, neg)

  # train SVM model (need to grid search for better parameters)
  # Check to do PCA
  # check to extract feature like HOG
  svmModel <- svm(formula = V1 ~ ., data = trainDataFinal, kernel= "radial", cost = 0.001, gamma = 0.1)
  predictAndEvaluateSVM(svmModel, trainDataFinal) # 0.50 :(
  predictAndEvaluateSVM(svmModel, valDataFinal)   # 0.50 :(
  
  # train NN model
  # (DOn't believe on this result . It have something wrong :) )
  nnModel = neuralnet(formula=f, data=trainDataFinal, hidden=c(5,3), linear.output=FALSE) 
  #linear.output = FALSE --> classification (apply 'logistic' activation as default)
  predictAndEvaluateNN(nnModel, trainDataFinal) #0.9792
  predictAndEvaluateNN(nnModel, valDataFinal)   #0.9953  

  # check how to save best model
  
  # check on confusion matrix if some number is more difficult to predict 
  # and train a special model for it
  
#}

# vooting method 
# ideia to pass 3 times to model and after majority voot to define the class

# final model prediction and confusion matrix on Training data
  
# final model prediction and confusion matrix on Validation data

# load test data
# Normalize test data (/255.0)
# final model prediction and confusion matrix on Test data
