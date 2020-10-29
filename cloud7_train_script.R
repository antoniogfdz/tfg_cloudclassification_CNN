library(RSNNS)
library(mxnet)
library(data.table)
library(foreach)

#require(mxnet)

# Cargar imagenes
meta <- readRDS(file="metadata.rds")#[,,,1,]
image <- readRDS(file="imageset.rds")#[,,,1,]

# añadir la columna
#test$clase<-x
train.x <- image[,,,1,]
#extrae la columna 23 (type)  de metadata
train.y <- meta[,23, drop=FALSE]

# ELIMINAR LAS CLASES:
# nimboestrato / aerosol / multinube
a1 <- which(train.y$types == 'nimboestrato')
b1 <- which(train.y$types == 'aerosol')
c1 <- which(train.y$types == 'multinube')
train.y <- train.y[-c(a1,b1,c1), ]
train.x <- train.x[ , , , -c(a1,b1,c1)]

# UNIFICAR LAS CLASES:
# cirro y cirroestrato
# cirrocumulo y altocumulo
# altoestrato y estrato

train.y[train.y=='cirroestrato']='cirro'
train.y[train.y=='cirrocumulo']='altocumulo'
train.y[train.y=='altoestrato']='estrato'

# 70% 15% 15%:
types <- unique(train.y)

for(type in types$types){
  assign(paste("type", type, sep = "."),which(train.y$types == type))
}

set.seed(1)

numberof <- c(
  length(type.altocumulo),
  length(type.cieloDespejado),
  length(type.cirro),
  length(type.cumulos),
  length(type.estrato),
  length(type.estratocumulo))

train.altocumulo<- sample(type.altocumulo, round(numberof[1]*0.7))
type.altocumulo <- setdiff(type.altocumulo, train.altocumulo)
train.cieloDespejado<- sample(type.cieloDespejado, round(numberof[2]*0.7))
type.cieloDespejado <- setdiff(type.cieloDespejado, train.cieloDespejado)
train.cirro<- sample(type.cirro, round(numberof[1]*0.7))
type.cirro <- setdiff(type.cirro, train.cirro)
train.cumulos<- sample(type.cumulos, round(numberof[4]*0.7))
type.cumulos <- setdiff(type.cumulos, train.cumulos)
train.estrato<- sample(type.estrato, round(numberof[5]*0.7))
type.estrato <- setdiff(type.estrato, train.estrato)
train.estratocumulo<- sample(type.estratocumulo, round(numberof[2]*0.7))
type.estratocumulo <- setdiff(type.estratocumulo, train.estratocumulo)

validation.altocumulo<- sample(type.altocumulo, round(numberof[1]*0.15))
test.altocumulo <- setdiff(type.altocumulo, validation.altocumulo)
validation.cieloDespejado<- sample(type.cieloDespejado, round(numberof[2]*0.15))
test.cieloDespejado <- setdiff(type.cieloDespejado, validation.cieloDespejado)
validation.cirro<- sample(type.cirro, round(numberof[1]*0.15))
test.cirro <- setdiff(type.cirro, validation.cirro)
validation.cumulos<- sample(type.cumulos, round(numberof[4]*0.15))
test.cumulos <- setdiff(type.cumulos, validation.cumulos)
validation.estrato<- sample(type.estrato, round(numberof[5]*0.15))
test.estrato <- setdiff(type.estrato, validation.estrato)
validation.estratocumulo<- sample(type.estratocumulo, round(numberof[2]*0.15))
test.estratocumulo <- setdiff(type.estratocumulo, validation.estratocumulo)

#RANDOM MIX & TRAIN / VALIDATION / TEST
aux <- c(
  train.altocumulo,
  train.cieloDespejado,
  train.cirro,
  train.cumulos,
  train.estrato,
  train.estratocumulo)

aux <- aux[sample(length(aux))]
trainset <- train.x[ , , , c(aux)]
classtrainset <- train.y[c(aux), ]
classtrainset$types=factor(classtrainset$types)
table(classtrainset)


aux <- c(
  validation.altocumulo,
  validation.cieloDespejado,
  validation.cirro,
  validation.cumulos,
  validation.estrato,
  validation.estratocumulo)

aux <- aux[sample(length(aux))]
validationset <- train.x[ , , , c(aux)]
classvalidationset <- train.y[c(aux), ]
classvalidationset$types=factor(classvalidationset$types)
table(classvalidationset)

aux <- c(
  test.altocumulo,
  test.cieloDespejado,
  test.cirro,
  test.cumulos,
  test.estrato,
  test.estratocumulo)

aux <- aux[sample(length(aux))]
testset <- train.x[ , , , c(aux)]
classtestset <- train.y[c(aux), ]
classtestset$types=factor(classtestset$types)
table(classtestset)

#Normalizacion
trainset <- trainset/255
validationset <- validationset/255
testset <- testset/255

channels <- dim(trainset)[1]
side <- dim(trainset)[2]
size <- dim(trainset)[4]
dim(trainset) <- c(channels,side,side,size)
trainset <- aperm(trainset,perm=c(2,3,1,4))

channels <- dim(validationset)[1]
side <- dim(validationset)[2]
size <- dim(validationset)[4]
dim(validationset) <- c(channels,side,side,size)
validationset <- aperm(validationset,perm=c(2,3,1,4))

channels <- dim(testset)[1]
side <- dim(testset)[2]
size <- dim(testset)[4]
dim(testset) <- c(channels,side,side,size)
testset <- aperm(testset,perm=c(2,3,1,4))

# PARÁMETROS DE LA RED
# num.round -> numero de ciclos (empezar con 1000)
# array.batch.size -> a partir de cuantos patrones se modifican los pesos (16,32)
# learning rate -> posiblemente baja (~10^(-5))
# momentum -> fijado a 0.9
# epoch.end.callback -> indica que se guardan los errores (metric) en cada ciclo (10, 20)
#                    -> depende del número de ciclos que se requiera finalmente.
# ------------- CAPAS DE CONVOLUCIÓN ------------------
# nº capas de convolucion -> empezamos con 1
# kernel -> dimensión de la ventana de convolución (3x3, 5x5)
# función de activación -> RELU
# pooling -> max / kernel (2x2) / stride (2x2)
# -------------------RED FINAL ------------------------
# nº de neuronas ocultas -> probar (nº entradas/2)
#                        ->intentar sacar el número de entradas que llegan a la red final

# nciclos <- c(1,500)
# batch_size <- c(16,32)
# learning_rate <- c(0.00001)
# momentum <- c(0.9)
# epoch <- c(10,20)
# krnel <- c(3,5)
# #func <- c(relu, tanh)
# #nocultas <- c((nº entradas/2))
#
# exp1 <- c(nciclos[1], batch_size[2])
# exp2 <- c(nciclos[2], batch_size[2])
# exp3 <- c(nciclos[1], batch_size[1])
#
# exp <- array(c(exp1,exp2,exp3),dim = c(2,3))
#
#
# listofmodels <- list()

#for (i in 1:1){ #ncol(exp)
# IMPORTANTE: se ha cambiado la clase de la salida cuando se llama al modelo. hay que poner y=as.numeric(classtrainset$types)

# #NETWORK
# input
data <- mx.symbol.Variable('data')

#pre-subsampling
pool0 <- mx.symbol.Pooling(data=data, pool_type="avg",
                           kernel=c(8,8), stride=c(8,8))

# first conv
conv1 <- mx.symbol.Convolution(data=pool0, kernel=c(5,5), num_filter=16)
relu1 <- mx.symbol.Activation(data=conv1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=relu1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=32)
relu2 <- mx.symbol.Activation(data=conv2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=relu2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=32) #182 = ceil(|trainset|/2)
relu3 <- mx.symbol.Activation(data=fc1, act_type="relu")
# # first fullc
# flatten <- mx.symbol.Flatten(data=pool2)
# fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=10)
# relu3 <- mx.symbol.Activation(data=fc1, act_type="relu")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=relu3, num_hidden=6)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

mx.set.seed(0)
tic <- proc.time()

log <- mx.metric.logger()
model <- mx.model.FeedForward.create(lenet, X=trainset, y=as.numeric(classtrainset$types)-1,
                                     #ctx=mx.gpu(),
                                     num.round=400,
                                     optimizer='adagrad',
                                     array.batch.size=8,
                                     #learning.rate=0.001, #momentum=0.9, wd=0.00001,
                                     eval.data= list(data= validationset,label=as.numeric(classvalidationset$types)-1),
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(10,log))

plot(log$train,type="l")

lines(log$eval,col="red")

trainTarget <- decodeClassLabels(unlist(classtrainset[,1,with=F]))
validTarget <- decodeClassLabels(unlist(classvalidationset[,1,with=F]))
testTarget <- decodeClassLabels(unlist(classtestset[,1,with=F]))

train.pred <- predict(model, trainset)
valid.pred <- predict(model, validationset)
test.pred <- predict(model, testset)

trainmatrix.pred <- t(as.matrix(train.pred))
#trainmatrix.pred <- apply(trainmatrix.pred, 2, which.max)
trainCm <- confusionMatrix(as.matrix(trainTarget), trainmatrix.pred)

validmatrix.pred <- t(as.matrix(valid.pred))
#validmatrix.pred <- apply(validmatrix.pred, 2, which.max)
validCm <- confusionMatrix(as.matrix(validTarget), validmatrix.pred)

testmatrix.pred <- t(as.matrix(test.pred))
#testmatrix.pred <- apply(testmatrix.pred, 2, which.max)
testCm <- confusionMatrix(as.matrix(testTarget), testmatrix.pred)

#METRICAS DE ERROR
accuracy <- function (cm) sum(diag(cm))/sum(cm)
#Sale warning cuando no clasifica alguna clase:
macroAvg <- function (cm) mean(diag(cm)/rowSums(cm))
accuracyPerClass <- function (cm) diag(cm)/rowSums(cm)


#VECTOR DE PRECISIONES
accuracies <- c(TrainAccuracy= accuracy(trainCm), TrainMacroAvg= macroAvg(trainCm), TrainAccuracyPerClass= accuracyPerClass(trainCm),
                ValidAccuracy= accuracy(validCm), ValidMacroAvg= macroAvg(validCm), ValidAccuracyPerClass= accuracyPerClass(validCm),
                TestAccuracy=  accuracy(testCm) , TestMacroAvg= macroAvg(testCm), TestAccuracyPerClass= accuracyPerClass(testCm))


#TABLA CON LOS ERRORES POR CICLO
#iterativeErrors <- data.table(RMSETrain= (model$IterativeFitError/nrow(trainset)) ^(1/2),
#                              RMSEValid= (model$IterativeTestError/nrow(testset))^(1/2))

# #UTILIDADES PARA GENERAR LAS TABLAS DE SALIDAS
# targets <- unlist(unique(trainSet[,target,with=F]))
predictClass <- function(predMat,targets) as.character(unlist(data.table(predMat)[,targets[apply(.SD,1,which.max)]]))
#
# #SALIDA DE LA RED EN BRUTO, SIN ELEGIR LA CLASE
# rawOutputsTrain <- trainPred
# rawOutputsTest  <- testPred
#
# #SALIDA DE LA RED ETIQUETADA (Eleccion del maximo por fila y asignar etiqueta)
outputsTrain <- predictClass(train.pred,classtrainset)
# outputsTest  <- predictClass(testPred,targets)

#
# RMSE  <- function(pred,obs) (sum((pred - obs) ^ 2) / length(obs)) ^ (1/2)
# nRMSE <- function(pred,obs,mn) RMSE(pred,obs) / mean(obs)
# MAE   <- function(pred,obs) sum(abs(pred - obs)) / length(obs)
# nMAE  <- function(pred,obs) MAE(pred,obs)/mean(obs)
#
#
# meanMAE <- c(train= MAE(mean(as.numeric(classtrainset$types)),as.numeric(classtrainset$types)),
#              valid= MAE(mean(as.numeric(classvalidationset$types)),as.numeric(classvalidationset$types)),
#              test= MAE(mean(as.numeric(classtestset$types)),as.numeric(classtestset$types)))
#
# meanRMSE <- c(train= RMSE(mean(as.numeric(classtrainset$types)),as.numeric(classtrainset$types)),
#               valid= RMSE(mean(as.numeric(classvalidationset$types)),as.numeric(classvalidationset$types)),
#               test= RMSE(mean(as.numeric(classtestset$types)),as.numeric(classtestset$types)))
#
# meanNRMSE <- c(train= nRMSE(mean(as.numeric(classtrainset$types)),as.numeric(classtrainset$types)),
#                valid= nRMSE(mean(as.numeric(classvalidationset$types)),as.numeric(classvalidationset$types)),
#                test= nRMSE(mean(as.numeric(classtestset$types)),as.numeric(classtestset$types)))
#
#
# predMAE <- c(train= MAE(train.pred,as.numeric(classtrainset$types)),
#              valid= MAE(valid.pred,as.numeric(classvalidationset$types)),
#              test= MAE(test.pred,as.numeric(classtestset$types)))
#
# predRMSE <- c(train= RMSE(train.pred,as.numeric(classtrainset$types)),
#               valid= RMSE(valid.pred,as.numeric(classvalidationset$types)),
#               test= RMSE(test.pred,as.numeric(classtestset$types)))
#
# predNRMSE <- c(train= nRMSE(train.pred,as.numeric(classtrainset$types)),
#                valid= nRMSE(valid.pred,as.numeric(classvalidationset$types)),
#                test= nRMSE(test.pred,as.numeric(classtestset$types)))

write.csv(train.pred, file = "train.csv")
write.csv(valid.pred, file = "valid.csv")
write.csv(test.pred, file = "test.csv")

prefix="6class_bloque1_2conv_16f_5s_32n"
iteration=400
mx.model.save(model, prefix, iteration)

print("done1")

#}
#end for
