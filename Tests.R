# See:
# https://cran.r-project.org/web/packages/keras/vignettes/index.html
# https://towardsdatascience.com/interpretable-machine-learning-for-image-classification-with-lime-ea947e82ca13

# Animals playing guitar and similar: https://w0286994blog.wordpress.com/

# keras installation: https://cran.r-project.org/web/packages/keras/vignettes/index.html
# install.packages("keras")
# library(keras)
# install_keras()

library(keras)
library(lime)
library(magick)
library(ciu.image)

# https://blogs.rstudio.com/ai/posts/2018-03-09-lime-v04-the-kitten-picture-edition/
lime.example <- function() {
  model <- application_vgg16(
    weights = "imagenet",
    include_top = TRUE
  )

  elephant <- image_read('elephant.jpg')

  print("Importing image")
  img <- image_read('https://www.data-imaginist.com/assets/images/kitten.jpg')
  #img <- elephant
  plot(as.raster(img)) # Plots in R's own plot system.
  img_path <- file.path(tempdir(), 'kitten.jpg')
  image_write(img, img_path)

  print("Building explainer")
  explainer <- lime(img_path, model, image_prep)

  print("Getting predictions")
  res <- predict(model, image_prep(img_path))
  imagenet_decode_predictions(res)
  # Redo explainer
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
  explainer <- lime(img_path, as_classifier(model, model_labels), image_prep)
  # Superpixel plot
  print("Plot superpixels")
  p <- plot_superpixels(img_path); print(p)
  # Changing some settings
  p <- plot_superpixels(img_path, n_superpixels = 200, weight = 40); print(p)
  print("Calling explain")
  explanation <- explain(img_path, explainer, n_labels = 3, n_features = 20) # This takes something like 5-6 minutes!
  explanation <- as.data.frame(explanation)
  # Default
  p <- plot_image_explanation(explanation); print(p)
  # Block out background
  p <- plot_image_explanation(explanation, display = 'block', threshold = 0.01); print(p)
  # Show negatively correlated areas as well
  p <- plot_image_explanation(explanation, threshold = 0, show_negative = TRUE, fill_alpha = 0.6) # Takes something like 15 seconds
  print(p)
}

#lime.example()

# MNIST. https://tensorflow.rstudio.com/guide/keras/
mnist.example <- function() {
  mnist <- dataset_mnist()
  x_train <- mnist$train$x
  y_train <- mnist$train$y
  x_test <- mnist$test$x
  y_test <- mnist$test$y
  # reshape
  x_train <- array_reshape(x_train, c(nrow(x_train), 784))
  x_test <- array_reshape(x_test, c(nrow(x_test), 784))
  # rescale
  x_train <- x_train / 255
  x_test <- x_test / 255
  y_train <- to_categorical(y_train, 10)
  y_test <- to_categorical(y_test, 10)
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 10, activation = 'softmax')
  summary(model)
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )
  history <- model %>% fit(
    x_train, y_train,
    epochs = 30, batch_size = 128,
    validation_split = 0.2
  )
  plot(history)
  print(model %>% evaluate(x_test, y_test))
  print(model %>% predict_classes(x_test))

  # CIU calculation.
  instance <- x_train[1,]
  n_pixels <- length(instance)
  ciu_x <- matrix(instance, nrow=n_pixels, ncol=n_pixels, byrow=TRUE)
  ciu_y <- predict(model, ciu_x)
  diag(ciu_x) <- 0
  ciu_zero <- predict(model, ciu_x)
  diag(ciu_x) <- 1
  ciu_one <- predict(model, ciu_x)
  diff <- ciu_one - ciu_zero
}
# mnist.example()

mnist.plot <- function() {
  mnist <- dataset_mnist()
  x_train <- mnist$train$x
  y_train <- mnist$train$y
  # visualize the digits
  par(mfcol=c(6,6))
  par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')
  for (idx in 1:36) {
    im <- x_train[idx,,]
    im <- t(apply(im, 2, rev))
    image(1:28, 1:28, im, col=gray((0:255)/255),
          xaxt='n', main=paste(y_train[idx]))
  }
  par(mfcol=c(1,1))
}

mnist.magick <- function() {
  mnist <- dataset_mnist()
  x_train <- mnist$train$x
  im <- x_train[1,,]
  img_grey  <- magick::image_read(im)
  im_raw <- img_grey[[1]]
  im_raw[2,,]<-im_raw[1,,]
  im_raw[3,,]<-im_raw[1,,]
  im_raw[4,,]<-as.raw(255)
  img <- magick::image_read(im_raw)
  img1<-image_rotate(img,90)
  img2<-image_flop(img1)
  plot(img2)
  imgfile <- tempfile()
  magick::image_write(img2, path = imgfile, format = 'png')
}

# https://cran.r-project.org/web/packages/magick/vignettes/intro.html
magick.tests <- function() {
  require(rsvg) # Only if using SVG images.
  tiger <- image_read_svg('http://jeroen.github.io/images/tiger.svg', width = 350)
  #print(tiger) # Opens own window, probably because of SVG
  #image_browse(tiger)

  elephant <- image_read('elephant.jpg')
  kitten <- image_read('https://www.data-imaginist.com/assets/images/kitten.jpg')
  img <- elephant
  print(img)
  img_path <- file.path(tempdir(), 'kitten.jpg')
  image_write(img, img_path)
  #p <- plot_superpixels(img_path); print(p)

  # From LIME source
  n_superpixels = 50
  weight = 20
  n_iter = 10
  p_remove = 0.5
  batch_size = 10
  background = 'grey'
  colour = 'black'
  #im <- magick::image_read(img_path)
  im_lab <- magick::image_convert(img, colorspace = 'LAB')
  super_pixels <- slic(
    magick::image_channel(im_lab, 'R')[[1]][1,,],
    magick::image_channel(im_lab, 'G')[[1]][1,,],
    magick::image_channel(im_lab, 'B')[[1]][1,,],
    n_sp = n_superpixels,
    weight = weight,
    n_iter = n_iter
  ) + 1

  # # plot_superpixels. Doesn't work with Tiger, for some reason.
  # contour <- magick::image_read(array(t(super_pixels)/max(super_pixels),
  #                                     dim = c(rev(dim(super_pixels)), 1)),
  #                               depth = 16)
  # #contour[contour != as.raw(0)] <- as.raw(255)
  # contour <- magick::image_convolve(contour, 'Laplacian')[[1]]
  # lines <- magick::image_blank(magick::image_info(im)$width, magick::image_info(im)$height, color = colour)
  # lines <- magick::image_convert(lines, type = 'TrueColorAlpha')[[1]]
  # lines[4,,] <- contour[1,,]
  # raster <- tidy_raster(magick::image_composite(im, magick::image_read(lines)))
  # ggplot(raster) +
  #   geom_raster(aes_(~x, ~y, fill = ~colour)) +
  #   coord_fixed(expand = FALSE) +
  #   scale_y_reverse() +
  #   scale_fill_identity() +
  #   theme_void()

  # Make transparent a given superpixel.
  im_raw <- magick::image_convert(img, type = 'TrueColorAlpha')[[1]]
  im_perm <- im_raw
  sp.ind <- c(28) # 28 is cat's back.
  im_perm[4,,][super_pixels %in% sp.ind] <- as.raw(0)
  im_perm <- magick::image_read(im_perm)
  im_perm <- magick::image_background(im_perm, background)
  print(im_perm)
  tmp <- tempfile()
  magick::image_write(im_perm, path = tmp, format = 'png')
  #unlink(tmp)

  # Get predictions
  model <- application_vgg16(
    weights = "imagenet",
    include_top = TRUE
  )

  # From here on, it doesn't work anymore! Spare code...

  # Original
  res <- predict(model, image_prep(img_path))
  origpreds <- imagenet_decode_predictions(res)

  # Perturbed
  decodes <- list()
  scores <- c()
  for ( i in 1:max(super_pixels) ) {
    tmp <- make.superpixels.transparent(img, c(i), super_pixels, background)
    res <- predict(model, image_prep(tmp))
    pert_preds <- imagenet_decode_predictions(res)
    decodes[[i]] <- pert_preds
    scores <- c(scores, pert_preds[[1]]$score[1])
    print(i)
    print(pert_preds)
    unlink(tmp)
  }

  # Plot with relevant superpixels only, rest as "background"
  im_perm <- im_raw
  plusmin <- origpreds[[1]]$score[1] - scores
  sp.ind <- which(plusmin<0.01)
  im_perm[4,,][super_pixels %in% sp.ind] <- as.raw(0)
  im_perm <- magick::image_read(im_perm)
  im_perm <- magick::image_background(im_perm, background)
  print(im_perm)
}

ciu_kitten <- function() {
  # Load image, write to local file.
  # img <- image_read('https://www.data-imaginist.com/assets/images/kitten.jpg')
  # img_path <- file.path(tempdir(), 'kitten.jpg')
  # image_write(img, img_path)
  img_path <- 'kitten.jpg'
  ciu_image(img_path, c(1,2,3), n_superpixels=100)
}

ciu_elephant <- function() {
  img_path <- 'elephant.jpg'
  ciu_image(img_path, c(1,2), threshold=0.05)
}

# See https://pbiecek.github.io/ema/LIME.html
ciu_DuckHorse <- function() {
  img_path <- 'HalfDuckHalfHorse_cropped.jpg'
  ciu_image(img_path, c(1,2), threshold=0.05)
}

ciu_image <- function(imgpath, ind.out=1, threshold = 0.02, n_superpixels=50) {
  # Get model if not initialized already
  if ( !exists("vgg16") || is.null(vgg16) )
    vgg16 <<- application_vgg16(
      weights = "imagenet",
      include_top = TRUE
    )
  # Create CIU object and get explanation.
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
  ciu <- ciu.image.new(vgg16, vgg_predict_function, output.names = model_labels)
  plist <- ciu$plot.image.explanation(imgpath, ind.output = ind.out, threshold = threshold,
                                      show_negative = FALSE, n_superpixels=50,
                                      weight=20, n_iter=10, background = 'grey')
  for ( i in 1:length(plist) )
    print(plist[[i]])
  res <- predict(vgg16, image_prep(imgpath))
  imagenet_decode_predictions(res)
}

ciu_LabradorGuitar <- function() {
  img_path <- 'LabradorPlayingGuitar_cropped.jpg'
  ciu_vgg19(img_path, c(1,2,3,4,5), threshold = 0.02)
}

ciu_vgg19 <- function(imgpath, ind.out=1, threshold = 0.02) {
  # Get model if not initialized already
  if ( !exists("vgg19") || is.null(vgg19) )
    vgg19 <<- application_vgg19(
      weights = "imagenet",
      include_top = TRUE
    )
  # Create CIU object and get explanation.
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
  ciu <- ciu.image.new(vgg19, vgg_predict_function, output.names = model_labels)
  plist <- ciu$plot.image.explanation(imgpath, ind.output = ind.out, threshold = threshold,
                                      show_negative = FALSE, n_superpixels=50,
                                      weight=20, n_iter=10, background = 'grey')
  for ( i in 1:length(plist) )
    print(plist[[i]])
  res <- predict(vgg19, image_prep(imgpath))
  imagenet_decode_predictions(res)
}

ciu_DogGuitar_InceptionV3 <- function(ind.out=1, threshold = 0.02, n_superpixels=50) {
  model <- application_inception_v3(weights = 'imagenet', include_top = TRUE)

  # Load image, write to local file.
  imgpath <- 'LabradorPlayingGuitar_cropped.jpg'
  res <- predict(model, image_prep_inception_v3(imgpath))
  imagenet_decode_predictions(res)
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
  ciu <- ciu.image.new(model, inc_v3_predict_function, output.names = model_labels)
  plist <- ciu$plot.image.explanation(imgpath, ind.output = ind.out, threshold = threshold,
                                      show_negative = FALSE, n_superpixels=n_superpixels,
                                      weight=20, n_iter=10, background = 'grey')
  for ( i in 1:length(plist) )
    print(plist[[i]])
}

ciu_DuckHorse_InceptionV3 <- function(ind.out=1, threshold = 0.02, n_superpixels=50) {
  model <- application_inception_v3(weights = 'imagenet', include_top = TRUE)

  # Load image, write to local file.
  imgpath <- 'HalfDuckHalfHorse_cropped.jpg'
  res <- predict(model, image_prep_inception_v3(imgpath))
  imagenet_decode_predictions(res)
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'ciu.image'))
  ciu <- ciu.image.new(model, inc_v3_predict_function, output.names = model_labels)
  plist <- ciu$plot.image.explanation(imgpath, ind.output = ind.out, threshold = threshold,
                                      show_negative = FALSE, n_superpixels=n_superpixels,
                                      weight=20, n_iter=10, background = 'grey')
  for ( i in 1:length(plist) )
    print(plist[[i]])
}


vgg_predict_function <- function(model, imgpath) {
  predict(model, image_prep(imgpath))
}

inc_v3_predict_function <- function(model, imgpath) {
  predict(model, image_prep_inception_v3(imgpath))
}

gastro_predict_function <- function(model, imgpath) {
  predict(model, gastro.image_prep(imgpath))
}

ciu_gastro <- function(imgpath, ind.out=c(1,2), threshold = 0.02, n_superpixels=50) {
  # Get model if not initialized already
  if ( !exists("gastromodel") || is.null(gastromodel) ) {
    #gastromodel <<- load_model_hdf5("~/Documents/Software/GastroImages/model_full_v24.h5")
    gastromodel <<- load_model_hdf5("~/Documents/Software/GastroImages/model_full_categorical.h5")
  }
  # Create CIU object and get explanation.
  ciu <- ciu.image.new(gastromodel, gastro_predict_function, output.names = c("Not Bleeding", "Bleeding"))
  plist <- ciu$plot.image.explanation(imgpath, ind.output = ind.out, threshold = threshold,
                                      show_negative = FALSE, n_superpixels=n_superpixels,
                                      weight=20, n_iter=10, background = 'grey')
  for ( i in 1:length(plist) )
    print(plist[[i]])
  res <- gastro_predict_function(gastromodel, imgpath)
}

lime_kitten <- function() {
  model <- application_vgg16(
    weights = "imagenet",
    include_top = TRUE
  )

  # Load image, write to local file.
  img <- image_read('https://www.data-imaginist.com/assets/images/kitten.jpg')
  img_path <- file.path(tempdir(), 'kitten.jpg')
  image_write(img, img_path)
  res <- predict(model, image_prep(img_path))
  imagenet_decode_predictions(res)
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
  explainer <- lime(img_path, as_classifier(model, model_labels), image_prep)
  explanation <- explain(img_path, explainer, n_labels = 3, n_features = 50, n_superpixels=50) # This takes something like 5-6 minutes!
  explanation <- as.data.frame(explanation)
  p <- plot_image_explanation(explanation); print(p)
  p <- plot_image_explanation(explanation, display = 'block', threshold = 0.01); print(p)
  p <- plot_image_explanation(explanation, threshold = 0, show_negative = TRUE, fill_alpha = 0.6) # Takes something like 15 seconds
  print(p)
}

lime_elephant <- function() {
  model <- application_vgg16(
    weights = "imagenet",
    include_top = TRUE
  )

  # Load image, write to local file.
  img_path <- 'elephant.jpg'
  res <- predict(model, image_prep(img_path))
  imagenet_decode_predictions(res)
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
  explainer <- lime(img_path, as_classifier(model, model_labels), image_prep)
  explanation <- explain(img_path, explainer, n_labels = 2, n_features = 20) # This takes something like 15 minutes!
  explanation <- as.data.frame(explanation)
  #p <- plot_image_explanation(explanation); print(p)
  p <- plot_image_explanation(explanation, display = 'block', threshold = 0.02); print(p)
  #p <- plot_image_explanation(explanation, threshold = 0, show_negative = TRUE, fill_alpha = 0.6) # Takes something like 15 seconds
  #print(p)
}

lime_DogGuitar <- function() {
  model <- application_vgg19(
    weights = "imagenet",
    include_top = TRUE
  )

  # Load image, write to local file.
  img_path <- 'LabradorPlayingGuitar_cropped.jpg'
  res <- predict(model, image_prep(img_path))
  imagenet_decode_predictions(res)
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
  explainer <- lime(img_path, as_classifier(model, model_labels), image_prep)
  explanation <- explain(img_path, explainer, n_labels = 5, n_features = 20) # This takes something like 15 minutes!
  explanation <- as.data.frame(explanation)
  #p <- plot_image_explanation(explanation); print(p)
  p <- plot_image_explanation(explanation, display = 'block', threshold = 0.01); print(p)
  #p <- plot_image_explanation(explanation, threshold = 0, show_negative = TRUE, fill_alpha = 0.6) # Takes something like 15 seconds
  #print(p)
}

lime_DogGuitar_InceptionV3 <- function() {
  model <- application_inception_v3(weights = 'imagenet', include_top = TRUE)

  # Load image, write to local file.
  img_path <- 'LabradorPlayingGuitar_cropped.jpg'
  res <- predict(model, image_prep_inception_v3(img_path))
  imagenet_decode_predictions(res)
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
  explainer <- lime(img_path, as_classifier(model, model_labels), image_prep_inception_v3)
  explanation <- explain(img_path, explainer, n_labels = 5, n_features = 20) # This takes something like 15 minutes!
  explanation <- as.data.frame(explanation)
  #p <- plot_image_explanation(explanation); print(p)
  p <- plot_image_explanation(explanation, display = 'block', threshold = 0.01); print(p)
  #p <- plot_image_explanation(explanation, threshold = 0, show_negative = TRUE, fill_alpha = 0.6) # Takes something like 15 seconds
  #print(p)
}

# See https://pbiecek.github.io/ema/LIME.html
lime_DuckHorse <- function() {
  model <- application_vgg16(
    weights = "imagenet",
    include_top = TRUE
  )

  # Load image, write to local file.
  img_path <- 'HalfDuckHalfHorse_cropped.jpg'
  res <- predict(model, image_prep(img_path))
  imagenet_decode_predictions(res)
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
  explainer <- lime(img_path, as_classifier(model, model_labels), image_prep)
  explanation <- explain(img_path, explainer, n_labels = 2, n_features = 100, n_superpixels = 100) # This takes something like 15 minutes!
  explanation <- as.data.frame(explanation)
  #p <- plot_image_explanation(explanation); print(p)
  p <- plot_image_explanation(explanation, display = 'block', threshold = 0.01); print(p)
  #p <- plot_image_explanation(explanation, threshold = 0, show_negative = TRUE, fill_alpha = 0.6) # Takes something like 15 seconds
  #print(p)
}

lime_DuckHorse_InceptionV3 <- function() {
  model <- application_inception_v3(weights = 'imagenet', include_top = TRUE)

  # Load image, write to local file.
  img_path <- 'HalfDuckHalfHorse_cropped.jpg'
  res <- predict(model, image_prep_inception_v3(img_path))
  imagenet_decode_predictions(res)
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))
  explainer <- lime(img_path, as_classifier(model, model_labels), image_prep_inception_v3)
  explanation <- explain(img_path, explainer, n_labels = 4, n_features = 50) # This takes something like 15 minutes!
  explanation <- as.data.frame(explanation)
  p <- plot_image_explanation(explanation, display = 'block', threshold = 0.01); print(p)
}

lime_gastro <- function(imgpath, threshold = 0.02, n_superpixels=50) {
  # Get model if not initialized already
  if ( !exists("gastromodel") || is.null(gastromodel) ) {
    #gastromodel <<- load_model_hdf5("~/Documents/Software/GastroImages/model_full_v24.h5")
    gastromodel <<- load_model_hdf5("~/Documents/Software/GastroImages/model_full_categorical.h5")
  }
  explainer <- lime(imgpath, as_classifier(gastromodel, c("Not Bleeding", "Bleeding")), gastro.image_prep)
  explanation <- explain(imgpath, explainer, n_labels = 2, n_features = n_superpixels, n_superpixels = n_superpixels) # This takes something like 15 minutes!
  explanation <- as.data.frame(explanation)
  p <- plot_image_explanation(explanation, display = 'block', threshold = threshold); print(p)
}

#lime_gastro("~/Documents/Software/GastroImages/dataset/Set 1/A/Set1_1.png")

LabradorGuitarVgg16 <- function() {
  fname <- 'LabradorPlayingGuitar.jpg'
  img <- image_read(fname)
  img_path <- file.path(tempdir(), fname)
  image_write(img, img_path)
  # model <- application_vgg16(
  #   weights = "imagenet",
  #   include_top = TRUE
  # )
  #model <- application_inception_v3()
  # model <- application_inception_resnet_v2(
  #   weights = "imagenet",
  #   include_top = TRUE
  # )
  model <- application_vgg19(
    weights = "imagenet",
    include_top = TRUE
  )
  res <- predict(model, image_prep(img_path))
  imagenet_decode_predictions(res, top=15)
}

LabradorGuitarInceptionV3 <- function() {
  model <- application_inception_v3(weights = 'imagenet', include_top = TRUE)
  #model <- InceptionV3(weights="imagenet")
  img_path <- 'LabradorPlayingGuitar_cropped.jpg'
  #img <- image_read(img_path)
  #img <- load_img(fname, target_size = c(299, 299))
  # x <- img_to_array(img)
  # x <- expand_dims(x, axis = 0)
  # x <- x / 255
  # res <- keras_predict(inception, x)
  # unlist(decode_predictions(pred, model = "InceptionV3", top = 10))
  res <- predict(model, image_prep_inception_v3(img_path))
  imagenet_decode_predictions(res, top=15)
}

image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(224,224))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

# Use this with inception v3 network
# Create network with "model <- application_inception_v3(weights = 'imagenet', include_top = TRUE)"
image_prep_inception_v3 <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(299,299))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- inception_v3_preprocess_input(x)
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

#' @importFrom grDevices as.raster
tidy_raster <- function(im) {
  if (!requireNamespace('magick', quietly = TRUE)) {
    stop('The magick package is required for image explanation', call. = FALSE)
  }
  raster <- as.raster(im)
  data.frame(x = rep(seq_len(ncol(raster)), nrow(raster)),
             y = rep(seq_len(nrow(raster)), each = ncol(raster)),
             colour = as.vector(raster),
             stringsAsFactors = FALSE)
}

test.gastroimages <- function() {
  require(keras)
  require(magick)
  require(lime)
  model <- load_model_hdf5("~/Documents/Software/GastroImages/model_full_v24.h5")
  predict(model, gastro.image_prep("dataset/Set 1/A/Set1_1.png"))
  plot_superpixels("dataset/Set 1/A/Set1_1.png")
  #img1_1 <- image_read("dataset/Set 1/A/Set1_1.png")
}

gastro.image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(150,150))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}


