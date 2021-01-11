\name{plot.image.explanation}
\alias{plot.image.explanation}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Plot CIU results for an image}
\description{Plot the image with CIU result visualisation.
\emph{"Usage" section is in
"Details" section because Roxygen etc. don't support documentation of functions
within functions.}
}
% "usage" section is not allowed, doesn't pass the devtools::check because
% there's no corresponding "top-level" function. So have to do it differently.
%\usage{
%explain(
%instance,
%ind.inputs.to.explain,
%in.min.max.limits=NULL,
%n.samples=100,
%target.concept=NULL,
%target.ciu=NULL
%)
%}
%- maybe also 'usage' for other objects documented here.
\details{

\bold{Usage}

\preformatted{
plot.image.explanation(
  imgpath,
  ind.outputs = 1,
  threshold = 0.02,
  show_negative = FALSE,
  n_superpixels=50,
  weight=20,
  n_iter=10,
  background = 'grey',
  strategy = "straight",
  ciu.sp.results = NULL,
  title = NULL)
}
}

\arguments{
  \item{imgpath}{Path to image file to explain. See \link{explain}.}
  \item{ind.outputs}{Indices of the outputs to calculate CIU for.}
  \item{threshold}{CI threshold, only show/emhphasize superpixels with
  CI>threshold.}
  \item{show_negative}{Include superpixels with "negative"
  (below "neutral" value) CU also. Default is FALSE. NOT IMPLEMENTED YET FOR "TRUE"!}
  \item{n_superpixels}{Number of superpixels to generate. See \link{explain}.}
  \item{weight}{See \link{explain}.}
  \item{n_iter}{See \link{explain}.}
  \item{background}{See \link{explain}.}
  \item{strategy}{See \link{ciu.superpixels}.}
  \item{ciu.sp.results}{Use these pre-calculated CIU values rather than
  re-calculating them. This is useful e.g. for creating a new plot with a
  different `threshold` value. This parameter can be the return value of a
  call to \code{\link{ciu.superpixels}} method or
  \code{\link{get.last.superpixel.ciu}} method right after calling
  \code{\link{plot.image.explanation}} for the image.
  See \link{ciu.superpixels}.}
  \item{title}{Replace default plot title with the value of this parameter.}
}

\value{A \code{\link{list}} of ggplot plots, one per output index in
\code{ind.outputs}.
}

\author{
  Kary Främling
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
