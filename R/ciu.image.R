#' ciu.image.new
#'
#' @param model The image recognitioning / classification model.
#' @param predict.function Function to call with image(s) as input and that produces
#' corresponding classification/output value [data.frame].
#' @param output.names Names of the classes that the outputs correspond to.
#'
#' @return A `ciu.image` object, whose methods can be called using `$` specifier.
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
  o.last.sp.ciu <- NULL

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
    tmp <- getfile.superpixels.transparent(o.image, ind.inputs.to.explain, o.super_pixels, background)
    #res <- predict(model, image_prep(tmp))
    pert.val <- as.vector(o.predict.function(o.model, tmp))
    unlink(tmp)

    minvals <- apply(matrix(c(cu.val,pert.val),ncol=2),1,min)
    maxvals <- apply(matrix(c(cu.val,pert.val),ncol=2),1,max)
    diff <- maxvals - minvals
    CI <- diff # Would normally be "/(o.absminmax[2] - o.absminmax[1])" but no point here for the moment since limited to [0,1] anyways.
    CU <- (cu.val - minvals)/diff
    CU[is.nan(CU)] <- 0.5 # If no diff, the zero CI and presumably undefined CU. But we set it to 0.5 for simplicity.

    # Finalize the return CIU object
    #o.last.ciu <<- ciu.image.result.new(CI, CU, minvals, maxvals, cu.val, origpreds[[1]]$class_description)
    o.last.ciu <<- ciu.image.result.new(CI, CU, minvals, maxvals, cu.val, o.outputnames)
    # Sort according to output value
    o.last.ciu <<- o.last.ciu[order(cu.val, decreasing=TRUE),]
    return(o.last.ciu)
  }

  # Get CIU for all superpixels in an image. Return value as a list of
  # ciu.image.result objects, one per superpixel.
  # "strategy" can take values "straight" (default) or "inverse" for the moment.
  ciu.superpixels <- function(imgpath, ind.outputs=1, n_superpixels=50,
                              weight=20, n_iter=10, background = 'grey',
                              strategy = "straight") {
    # Verify that we have image and superpixels
    check.superpixels(imgpath, n_superpixels, weight, n_iter)
    n.spixels <- max(o.super_pixels)
    all.sp <- seq(1,n.spixels)
    CIs <- CUs <- cmins <- cmaxs <- matrix(0, nrow=length(ind.outputs), ncol=n.spixels)
    for ( i in 1:n.spixels ) {
      if ( strategy == "inverse" )
        inps <- all.sp[-i]
      else
        inps <- c(i)
      ciu <- explain(imgpath, ind.inputs.to.explain=inps,
                     n_superpixels=n_superpixels, weight=weight,
                     n_iter=n_iter, background=background)
      CIs[,i] <- ciu[ind.outputs,]$CI
      CUs[,i] <- ciu[ind.outputs,]$CU
      cmins[,i] <- ciu[ind.outputs,]$cmin
      cmaxs[,i] <- ciu[ind.outputs,]$cmax
    }
    o.last.sp.ciu <<- list(out.names=ciu$out.names[ind.outputs], outval=ciu$outval[ind.outputs],
                           CI=CIs, CU=CUs, cmin=cmins, cmax=cmaxs)
    return(o.last.sp.ciu)
  }

  # Find "best" number of superpixels from a CI point of view. Don't know yet
  # which indicator is the best one, enabling max and sum for the moment.
  find.best.n.superpixels <- function(imgpath, ind.output=1,
                                      min_superpixels=5, max_superpixels=50,
                                      weight=20, n_iter=10, background = 'grey') {
    if ( is.null(o.last.imgpath) || o.last.imgpath != imgpath ) {
      o.image <<- magick::image_read(imgpath)
      o.last.imgpath <<- imgpath
    }

    maxCI <- 0
    ind.maxCI <- min_superpixels
    nbr.decreased.CI <- 0
    for ( n_pixels in min_superpixels:max_superpixels ) {
      # Create superpixels here.
      o.super_pixels <<- create.superpixels(o.image, n_pixels, weight, n_iter)
      ciu.sp <- ciu.superpixels(imgpath, ind.outputs=ind.output, n_superpixels=n_pixels,
                                weight=weight, n_iter=n_iter, background=background)
      # mCI <- max(ciu.sp$CI)
      mCI <- sum(ciu.sp$CI)
      if ( mCI >= maxCI ) { # ">=", equal is important because we presumably prefer higher precision.
        maxCI <- mCI
        ind.maxCI <- n_pixels # Number of superpixels is not necessarily as many as we asked for.
        nbr.decreased.CI <- 0
      }
      else {
        nbr.decreased.CI <- nbr.decreased.CI + 1
      }
      if ( nbr.decreased.CI > 2 ) # Detect if results are getting worse.
        break
    }
    return(ind.maxCI)
  }

  # - ciu.sp.results If CIU has already been calculated for this image and
  # superpixels, then they can be provided rather than re-calculated (saves time).
  plot.image.explanation <- function(imgpath, ind.outputs = 1, threshold = 0.02,
                                     show_negative = FALSE, n_superpixels=50,
                                     weight=20, n_iter=10, background = 'grey',
                                     strategy = "straight", ciu.sp.results = NULL,
                                     title = NULL) {
    # Verify that we have image and superpixels
    check.superpixels(imgpath, n_superpixels, weight, n_iter)

    # Get CIU for all superpixels, unless given as parameter
    if ( is.null(ciu.sp.results) )
      ciu.sp <- ciu.superpixels(imgpath, ind.outputs=ind.outputs, strategy=strategy)
    else
      ciu.sp <- ciu.sp.results

    # One plot per output to explain
    plist <- list()
    for ( i in 1:length(ind.outputs) ) {
      CIs <- ciu.sp$CI[i,]
      CUs <- ciu.sp$CU[i,]
      if (strategy == "inverse")
        CIs <- 1 - CIs
      # Plot with relevant superpixels only, rest as "background"
      if ( show_negative )
        sp.ind <- which(CIs < threshold) # This doesn't make sense yet, without color masking.
      else
        sp.ind <- which((CIs < threshold) | (CUs <= 0.5)) # 0.5 is "neutral" CU but that hardly matters here.
      # tmp <- getfile.superpixels.transparent(o.image, sp.ind, o.super_pixels, background)
      # img <- magick::image_read(tmp)
      img <- make.superpixels.transparent(o.image, sp.ind, o.super_pixels, background)
      if ( is.null(title) )
        t <- paste0(ciu.sp$out.names[i], ", probability: ",
                    format(ciu.sp$outval[i], digits=3), "\nCI threshold=", threshold,
                    ", #superpixels=", n_superpixels)
      plist[[i]] <- magick::image_ggplot(img) +
        ggtitle(t)
    }
    return(plist)
  }

  # Create ciu plots with number of superpixels beginning with
  # "start_superpixels" and ending with "end_superpixels".
  plot.superpixels.sequence <- function(imgpath, ind.output = 1, threshold = 0.02,
                                        show_negative = FALSE,
                                        start_superpixels=2, end_superpixels=50,
                                        weight=20, n_iter=10, background = 'grey') {
    # We want to read the image only once
    if ( is.null(o.last.imgpath) || o.last.imgpath != imgpath ) {
      o.image <<- magick::image_read(imgpath)
      o.last.imgpath <<- imgpath
    }

    # One plot per output to explain
    plist <- list()
    for ( i in start_superpixels:end_superpixels ) {
      o.super_pixels <<- create.superpixels(o.image, i, weight, n_iter)
      pl <- plot.image.explanation(imgpath, ind.outputs = ind.output,
                                   threshold = threshold,
                                   show_negative = show_negative, n_superpixels=i,
                                   weight=weight, n_iter=n_iter, background = background)
      plist[[i]] <- pl[[1]]
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

  # # Make transparent the given superpixel(s) and write to temporary file.
  # getfile.superpixels.transparent <- function(img, sp.ind, super_pixels, background) {
  #   im_perm <- make.superpixels.transparent(img, sp.ind, super_pixels, background)
  #   tmp <- tempfile()
  #   magick::image_write(im_perm, path = tmp, format = 'png')
  #   return(tmp)
  # }

  # Make transparent the given superpixel(s).
  make.superpixels.transparent <- function(img, sp.ind, super_pixels, background) {
    im_raw <- magick::image_convert(img, type = 'TrueColorAlpha')[[1]]
    im_perm <- im_raw
    im_perm[4,,][super_pixels %in% sp.ind] <- as.raw(0)
    im_perm <- magick::image_read(im_perm)
    im_perm <- magick::image_background(im_perm, background)
    return(im_perm)
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
                               weight=20, n_iter=10, background = 'grey',
                               strategy = "straight") {
      ciu.superpixels(imgpath, ind.outputs, n_superpixels,
                      weight, n_iter, background, strategy)
    },
    find.best.n.superpixels = function(imgpath, ind.output=1,
                                       min_superpixels=5, max_superpixels=50,
                                       weight=20, n_iter=10, background = 'grey') {
      find.best.n.superpixels(imgpath, ind.output, min_superpixels, max_superpixels,
                              weight, n_iter, background )
    },
    plot.image.explanation = function(imgpath, ind.outputs = 1, threshold = 0.02,
                                      show_negative = FALSE, n_superpixels=50,
                                      weight=20, n_iter=10, background = 'grey',
                                      strategy = "straight", ciu.sp.results = NULL,
                                      title = NULL) {
      plot.image.explanation(imgpath, ind.outputs, threshold,
                             show_negative, n_superpixels,
                             weight, n_iter, background, strategy,
                             ciu.sp.results, title)
    },
    plot.superpixels.sequence = function(imgpath, ind.output = 1, threshold = 0.02,
                                         show_negative = FALSE,
                                         start_superpixels=2, end_superpixels=50,
                                         weight=20, n_iter=10, background = 'grey') {
      plot.superpixels.sequence(imgpath, ind.output, threshold, show_negative,
                                start_superpixels, end_superpixels,
                                weight, n_iter, background)
    },
    reset.superpixels = function() o.super_pixels <<- NULL,
    get.super_pixels = function() o.super_pixels,
    get.last.superpixel.ciu = function() o.last.sp.ciu
  )

  class(this) <- c("ciu.image", class(this))
  return(this)
}
