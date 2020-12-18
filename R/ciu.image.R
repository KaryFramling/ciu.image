#' ciu.image.new
#'
#' @param model blabla
#' @param predict.function blabla
#' @param output.names blabla
#'
#' @return blabla
#' @export
#' @import ggplot2
#' @importFrom stats predict
#' @importFrom keras imagenet_decode_predictions image_load
#' image_to_array array_reshape imagenet_preprocess_input
#' @importFrom lime slic
#' @importFrom magick image_ggplot
ciu.image.new <- function(model, predict.function=NULL, output.names=NULL) {

  o.model <- model
  o.outputnames <- output.names
  o.absminmax <- matrix(c(0,1), ncol=2, byrow=TRUE)
  o.super_pixels <- NULL
  o.last.ciu <- NULL
  o.last.imgpath <- NULL
  o.image <- NULL

  # Deal with default models here, such as VGG, Inception_V3 etc.
  o.predict.function <- predict.function
  #   if ( is.null(o.predict.function) ) {
  # }

  explain <- function(imgpath, ind.inputs.to.explain=c(1), in.min.max.limits=NULL,
                      n.samples=100, n_superpixels=50, weight=20, n_iter=10,
                      background = 'grey',
                      target.concept=NULL, target.ciu=NULL) {

    # Verify that we have image and superpixels
    check.superpixels(imgpath, n_superpixels, weight, n_iter)

    # Original
    cu.val <- as.vector(o.predict.function(o.model, imgpath))

    # Perturbed image.
    tmp <- make.superpixels.transparent(o.image, ind.inputs.to.explain, o.super_pixels, background)
    #res <- predict(model, image_prep(tmp))
    pert.val <- as.vector(o.predict.function(o.model, tmp))

    minvals <- apply(matrix(c(cu.val,pert.val),ncol=2),1,min)
    maxvals <- apply(matrix(c(cu.val,pert.val),ncol=2),1,max)
    diff <- maxvals - minvals
    CI <- diff # Would normally be "/(o.absminmax[2] - o.absminmax[1])" but no point here for the moment since limited to [0,1] anyways.
    CU <- (cu.val - minvals)/diff
    is.na(CU) <- 0.5 # If no diff, the zero CI and presumably undefined CU. But we set it to 0.5 for simplicity.

    # Finalize the return CIU object
    #o.last.ciu <<- ciu.image.result.new(CI, CU, minvals, maxvals, cu.val, origpreds[[1]]$class_description)
    o.last.ciu <<- ciu.image.result.new(CI, CU, minvals, maxvals, cu.val, o.outputnames)
    # Sort according to output value
    o.last.ciu <<- o.last.ciu[order(cu.val, decreasing=TRUE),]
    return(o.last.ciu)
  }

  # explain <- function(imgpath, ind.inputs.to.explain=c(1), in.min.max.limits=NULL,
  #                     n.samples=100, n_superpixels=50, weight=20, n_iter=10,
  #                     background = 'grey',
  #                     target.concept=NULL, target.ciu=NULL) {
  #
  #   # Verify that we have image and superpixels
  #   check.superpixels(imgpath, n_superpixels, weight, n_iter)
  #
  #   # Original
  #   origres <- o.predict.function(o.model, imgpath)
  #
  #   # Perturbed image.
  #   tmp <- make.superpixels.transparent(o.image, ind.inputs.to.explain, o.super_pixels, background)
  #   #res <- predict(model, image_prep(tmp))
  #   pertres <- o.predict.function(o.model, tmp)
  #
  #   # Special treatment for imagenet_decode_predictions here. Not sure if it's
  #   # the best way but will have to do for the moment.
  #   #    if ( inherits(o.model, "keras.engine.training.Model" )) {
  #   if ( ncol(origres) == 1000 ) {
  #     origpreds <- imagenet_decode_predictions(origres, top=length(origres))
  #     pert_preds <- imagenet_decode_predictions(pertres, top=length(pertres))
  #
  #     # Merge data frames together using class_name as common denominator
  #     joint_preds <- merge(origpreds[[1]],pert_preds[[1]], by="class_name",sort=FALSE)
  #
  #     # Seems like we have the order that we want but "merge" doesn't guarantee
  #     # any order according to documentation. So we do a sort, just in case.
  #     joint_preds <- joint_preds[order(joint_preds$score.x,decreasing = TRUE),]
  #     cu.val <- joint_preds$score.x
  #     pert.val <- joint_preds$score.y
  #   }
  #   else {
  #     cu.val <- origres
  #     pert.val <- pertres
  #   }
  #   minvals <- apply(matrix(c(cu.val,pert.val),ncol=2),1,min)
  #   maxvals <- apply(matrix(c(cu.val,pert.val),ncol=2),1,max)
  #   diff <- maxvals - minvals
  #   CI <- diff # Would normally be "/(o.absminmax[2] - o.absminmax[1])" but no point here for the moment since limited to [0,1] anyways.
  #   CU <- (cu.val - minvals)/diff
  #
  #   # Finalize the return CIU object
  #   o.last.ciu <<- ciu.image.result.new(CI, CU, minvals, maxvals, cu.val, origpreds[[1]]$class_description)
  #   return(o.last.ciu)
  # }

  # Get CIU for all superpixels
  ciu.superpixels <- function(imgpath, ind.outputs=1, n_superpixels=50,
                              weight=20, n_iter=10, background = 'grey') {
    # Verify that we have image and superpixels
    check.superpixels(imgpath, n_superpixels, weight, n_iter)
    n.spixels <- max(o.super_pixels)
    CIs <- CUs <- cmins <- cmaxs <- matrix(0, nrow=length(ind.outputs), ncol=n.spixels)
    for ( i in 1:n.spixels ) {
      ciu <- explain(imgpath, ind.inputs.to.explain=c(i),
                     n_superpixels=n_superpixels, weight=weight,
                     n_iter=n_iter, background=background)
      CIs[,i] <- ciu[ind.outputs,]$CI
      CUs[,i] <- ciu[ind.outputs,]$CU
      cmins[,i] <- ciu[ind.outputs,]$cmin
      cmaxs[,i] <- ciu[ind.outputs,]$cmax
    }
    CIUs <- list(out.names=ciu$out.names[ind.outputs], outval=ciu$outval[ind.outputs],
                 CI=CIs, CU=CUs, cmin=cmins, cmax=cmaxs)
    return(CIUs)
  }

  plot.image.explanation <- function(imgpath, ind.outputs = 1, threshold = 0.02,
                                     show_negative = FALSE, n_superpixels=50,
                                     weight=20, n_iter=10, background = 'grey') {
    # Verify that we have image and superpixels
    check.superpixels(imgpath, n_superpixels, weight, n_iter)

    # Get CIU for all superpixels
    ciu.sp <- ciu.superpixels(imgpath, ind.outputs=ind.outputs)

    # One plot per output to explain
    plist <- list()
    for ( i in 1:length(ind.outputs) ) {
      CIs <- ciu.sp$CI[i,]
      CUs <- ciu.sp$CU[i,]
      # Plot with relevant superpixels only, rest as "background"
      if ( show_negative )
        sp.ind <- which(CIs < threshold) # This doesn't make sense yet, without color masking.
      else
        sp.ind <- which((CIs < threshold) | (CUs < 0.5)) # 0.5 is "neutral" CU but that hardly matters here.
      tmp <- make.superpixels.transparent(o.image, sp.ind, o.super_pixels, background)
      img <- magick::image_read(tmp)
      # print(img) # No labesl etc possible here
      #plot(as.raster(img), main=paste(ciu[ind.output]$out.names, "Prob:", ciu[ind.output]$cu.val))
      plist[[i]] <- magick::image_ggplot(img) +
        ggtitle(paste0(ciu.sp$out.names[i], ", probability: ",
                       format(ciu.sp$outval[i], digits=3), ", CI threshold=", threshold))
    }
    return(plist)
  }

  # Create superpixels segmentation if it doesn't exist or if image has
  # changed.
  check.superpixels <- function(imgpath, n_superpixels=50, weight=20, n_iter=10) {
    if ( is.null(o.super_pixels) || is.null(o.last.imgpath) || imgpath != o.last.imgpath ) {
      o.image <<- magick::image_read(imgpath)
      o.super_pixels <<- create.superpixels(o.image, n_superpixels, weight, n_iter)
    }
    o.last.imgpath <<- imgpath
  }

  create.superpixels <- function(img, n_superpixels=50, weight=20, n_iter=10) {
    im_lab <- magick::image_convert(img, colorspace = 'LAB')
    super_pixels <- slic(
      magick::image_channel(im_lab, 'R')[[1]][1,,],
      magick::image_channel(im_lab, 'G')[[1]][1,,],
      magick::image_channel(im_lab, 'B')[[1]][1,,],
      n_sp = n_superpixels,
      weight = weight,
      n_iter = n_iter
    ) + 1
    return(super_pixels)
  }

  # This is now specific for ImageNet images (224x224)
  # image_prep <- function(x) {
  #   arrays <- lapply(x, function(path) {
  #     img <- image_load(path, target_size = c(224,224))
  #     x <- image_to_array(img)
  #     x <- array_reshape(x, c(1, dim(x)))
  #     x <- imagenet_preprocess_input(x)
  #   })
  #   do.call(abind::abind, c(arrays, list(along = 1)))
  # }

  # Make transparent the given superpixel(s).
  make.superpixels.transparent <- function(img, sp.ind, super_pixels, background) {
    im_raw <- magick::image_convert(img, type = 'TrueColorAlpha')[[1]]
    im_perm <- im_raw
    im_perm[4,,][super_pixels %in% sp.ind] <- as.raw(0)
    im_perm <- magick::image_read(im_perm)
    im_perm <- magick::image_background(im_perm, background)
    tmp <- tempfile()
    magick::image_write(im_perm, path = tmp, format = 'png')
    return(tmp)
  }

  # Return list of "public" methods
  this <- list(
    explain = function(imgpath, ind.inputs.to.explain=c(1), in.min.max.limits=NULL,
                       n.samples=100, n_superpixels=50, weight=20, n_iter=10,
                       background = 'grey',
                       target.concept=NULL, target.ciu=NULL) {
      explain(imgpath, ind.inputs.to.explain, in.min.max.limits,
              n.samples, n_superpixels, weight, n_iter,
              background,
              target.concept, target.ciu)
    },
    ciu.superpixels = function(imgpath, ind.outputs=1, n_superpixels=50,
                               weight=20, n_iter=10, background = 'grey') {
      ciu.superpixels(imgpath, ind.outputs, n_superpixels,
                      weight, n_iter, background)
    },
    plot.image.explanation = function(imgpath, ind.outputs = 1, threshold = 0.02,
                                      show_negative = FALSE, n_superpixels=50,
                                      weight=20, n_iter=10, background = 'grey') {
      plot.image.explanation(imgpath, ind.outputs, threshold,
                             show_negative, n_superpixels,
                             weight, n_iter, background)
    },
    get.super_pixels = function() o.super_pixels
  )

  class(this) <- c("ciu.image", class(this))
  return(this)
}
