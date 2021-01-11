# ciu.image
R implementation of Contextual Importance and Utility for Explainable AI with Images.

# Background

CIU was developed by Kary Främling in his PhD thesis *Learning and Explaining Preferences with Neural Networks for Multiple Criteria Decision Making*, (written in French, title *Modélisation et apprentissage des préférences par réseaux de neurones pour l'aide à la décision multicritère*), available online for instance here: https://tel.archives-ouvertes.fr/tel-00825854/document. It was originally implemented in Matlab and has later been re-implemented in Python and R (package `ciu`) for tabular data. 

This `ciu.image` package implements CIU for image recognition "explanation" using e.g. saliency maps. 


# Installation

In the future, ciu.image will presumably be available from CRAN. Meanwhile, it can be installed directly from Github with the command 

```
# install.packages('devtools') # Uncomment if devtools wasn't installed already
devtools::install_github('KaryFramling/ciu.image')
```

# Running

The root directory contains several source files with example code for using `ciu.image`. A simple example using a kitten image with VGG16 is shown here. 

``` r
library(keras)
library(lime)
library(magick)
library(ciu.image)

vgg_predict_function <- function(model, imgpath) {
  predict(model, image_prep(imgpath))
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

model <- application_vgg16(
    weights = "imagenet",
    include_top = TRUE
  )
model_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package='ciu.image'))
imgpath <- system.file('extdata', 'kitten.jpg', package = 'ciu.image')
ciu <- ciu.image.new(model, vgg_predict_function, output.names = model_labels)
plist <- ciu$plot.image.explanation(imgpath, ind.output = c(1,2,3))
for ( i in 1:length(plist) )
  print(plist[[i]])

```

# Author

[Kary Främling](http://github.com/KaryFramling)

# References
The first publication on CIU was in the ICANN conference in Paris in 1995: *FRÄMLING, Kary, GRAILLOT, Didier. Extracting Explanations from Neural Networks. ICANN'95 proceedings, Vol. 1, Paris, France, 9-13 October, 1995. Paris: EC2 & Cie, 1995. pp. 163-168.*, accessible at http://www.cs.hut.fi/u/framling/Publications/FramlingIcann95.pdf.

The second publication, and last before "hibernation" of CIU research, is *FRÄMLING, Kary. Explaining Results of Neural Networks by Contextual Importance and Utility. Proceedings of the AISB'96 conference, 1-2 April 1996. Brighton, UK, 1996.*, accessible at http://www.cs.hut.fi/u/framling/Publications/FramlingAisb96.pdf.

The first publication after "hibernation" is *ANJOMSHOAE, Sule, FRÄMLING, Kary, NAJJAR, Amro. Explanations of black-box model predictions by contextual importance and utility. In: Lecture Notes in Computer Science, Vol. 11763 LNAI; Revised Selected Papers of Explainable, Transparent Autonomous Agents and Multi-Agent Systems - 1st International Workshop, EXTRAAMAS 2019, Montreal, Canada, May 13-14, 2019. pp. 95-109.* https://www.researchgate.net/publication/333310978_Explanations_of_Black-Box_Model_Predictions_by_Contextual_Importance_and_Utility.




