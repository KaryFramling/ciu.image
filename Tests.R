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
library(ggplot2)
library(ciu.image)

source("ImageHelpers.R")

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

plot.image.explanation.test <- function(imgpath, model, ind.outputs = 1, threshold = 0.02,
                                       show_negative = FALSE, n_superpixels=50,
                                       weight=20, n_iter=10, background = 'grey') {
  img <- image_read(imgpath)
  sp <- create.superpixels(img, n_superpixels = n_superpixels)
  n_pixels <- max(sp)
  all.sp <- seq(1,n_pixels)
  model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'ciu.image'))
  ciu <- ciu.image.new(model, inc_v3_predict_function, model_labels)
  sp.res <- data.frame(CI=c(), CU=c(), sp.len=c())
  sp.list <- list()
  for ( i in 1:n_pixels ) {
    sp.ind <- all.sp[-i]
    perm <- make.superpixels.transparent(img, sp.ind, sp, "grey")
    #print(magick::image_ggplot(perm) + ggtitle(paste("Superpixel #", i)))
    ciu.res <- ciu$explain(imgpath, ind.inputs.to.explain=sp.ind, n_superpixels=n_superpixels)
    #print(ciu.res)
    ciu.outp <- ciu.res[ind.outputs,]
    sp.res <- rbind(sp.res, data.frame(CI=ciu.outp$CI, CU=ciu.outp$CU, sp.len=length(sp.ind)))
    sp.list[[length(sp.list)+1]] <- all.sp[-i]
  }
  ci.order <- order(sp.res$CI)
  sp.res <- sp.res[ci.order,]
  sp.list <- sp.list[ci.order]
  n.sp.to.show <- sum((sp.res$CI <= 0.99)) # & (sp.res$CU > 0.5) )
  n.sp.to.show <- max(1, n.sp.to.show)
  n.sp.to.show <- min(length(ci.order), n.sp.to.show)
  perm <- make.superpixels.transparent(img, ci.order[(n.sp.to.show+1):length(ci.order)], sp, "grey")
  p <- magick::image_ggplot(perm) + ggtitle(paste("Image:", basename(imgpath)))
}
