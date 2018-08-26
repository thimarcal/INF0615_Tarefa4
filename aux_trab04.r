# Exemplo de leitura do dataset
# A primeira feature/coluna (V1) é a classe 
# As colunas V2 ~ V785 são os pixels de cada imagem 28x28 em tons de cinza. 
data = read.csv("mnist_trainVal.csv", header=FALSE)

#Inspecionando o número de exemplos por classe
summary(as.factor(data[,1]))



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