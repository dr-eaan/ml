setwd('C:\\Users\\proto\\Desktop\\Iacus project')
library(dplyr)
library(keras)
library(tensorflow)

rm(list=ls()[! ls() %in% c('trpicol', 'tspicol')])
gc()
set.seed(42)
trpicol = readRDS('trcolfull.rds')
tspicol = readRDS('tscolfull.rds')
trpicol = readRDS('trcolbal.rds')
tspicol = readRDS('tscolbal.rds')
trpicol = readRDS('trcol4000.rds')
tspicol = readRDS('tscol1000.rds')

w = dim(trpicol[[1]])[1] * dim(trpicol[[1]])[2] * dim(trpicol[[1]])[3] #all have the same dimensions
x_train = matrix(0, nrow = length(trpicol), ncol = w)
x_test = matrix(0, nrow = length(tspicol), ncol = w)

z = 0
for (i in 1:length(trpicol)){
  z = z + 1
  x = 1
  for (s in 1:dim(trpicol[[i]])[3]){
    for (r in 1:dim(trpicol[[i]])[1]){
      for (c in 1:dim(trpicol[[i]])[2]){
        x_train[z, x] = trpicol[[i]][r, c, s]
        x = x + 1
      }
    }
  }
}

saveRDS(x_train, file = 'x_traincnnfull.rds')

z = 0
for (i in 1:length(tspicol)){
  z = z + 1
  x = 1
  for (s in 1:dim(tspicol[[i]])[3]){
    for (r in 1:dim(tspicol[[i]])[1]){
      for (c in 1:dim(tspicol[[i]])[2]){
        x_test[z, x] = tspicol[[i]][r, c, s]
        x = x + 1
      }
    }
  }
}

saveRDS(x_test, file = 'x_testcnnfull.rds')
rm(trpicol, tspicol)
gc()

x_train = array(x_train, c(nrow(x_train), 50, 50, 3))
x_test = array(x_test, c(nrow(x_test), 50, 50, 3))
y_train = to_categorical(readRDS('tridhogfull.rds'))
y_test = to_categorical(readRDS('tsidhogfull.rds'))
y_train= to_categorical(readRDS('tridhogbal.rds'))
y_test = to_categorical(readRDS('tsidhogbal.rds'))
y_train = to_categorical(readRDS('tridhog4000.rds'))
y_test = to_categorical(readRDS('tsidhog1000.rds'))

es <- keras::callback_early_stopping(monitor = "val_loss",
                                      verbose = 1,
                                      patience = 5,
                                      min_delta = 0.01)
rm(list=ls()[! ls() %in% c('x_train', 'x_test', 'y_train', 'y_test')])

model = keras_model_sequential() %>% 
  layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = "relu", input_shape = c(50, 50, 3)) %>%
    layer_max_pooling_2d(pool_size = c(4, 4)) %>%
      layer_conv_2d(filters = 64, kernel_size = c(2, 2), activation = "relu") %>%
        layer_flatten() %>%
          layer_dropout(rate = 0.5) %>%
            layer_dense(units = 300, activation = "relu") %>%
              layer_dropout(0.2) %>%
              layer_dense(units = 200, activation = "tanh") %>%
                layer_dropout(0.2) %>%
                layer_dense(units = dim(y_train)[2], activation = "softmax")
model %>% compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')
t1 = Sys.time()
history = model %>% fit(x_train, y_train, epochs = 50, batch_size = 50) #, validation_split = 0.2,  callbacks = c(es)
t2 = Sys.time()
cnntime = t2 - t1 #5.791336 min for full with early stop, 29.75018 min for full, 6.139876 for 4000, 10.2203 for bal
model %>% evaluate(x_test, y_test)
#98.49% train, 82.46% test (4000), 99.59% train 73.93% test (bal), 98.82% train 85.94% test for 4000, 96.59% train 73.89% test (full with val), 99.25% train 93.48% test full