\name{ciu.superpixels}
\alias{ciu.superpixels}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Calculate CIU for all superpixels of an image}
\description{Calculate Contextual Importance (CI) and Contextual Utility (CU)
for all superpixels of an image (Context) using the given "black-box" model.
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
ciu.superpixels(
  imgpath,
  ind.outputs=1,
  n_superpixels=50,
  weight=20,
  n_iter=10,
  background = 'grey',
  strategy = "straight"
)
}
}

\arguments{
  \item{imgpath}{Path to image file to explain. See \link{explain}.}
  \item{ind.outputs}{Indices of the outputs to calculate CIU for.}
  \item{n_superpixels}{Number of superpixels to generate. See \link{explain}.}
  \item{weight}{See \link{explain}.}
  \item{n_iter}{See \link{explain}.}
  \item{background}{See \link{explain}.}
  \item{strategy}{Can be 'straight' or 'inverse'. 'straight' signifies varying
  the value of the studied pixel only. 'inverse' changes value of all superpixels,
  except the one studied. For the moment, the only value change used is to set
  superpixel(s) to transparent or not.}
}

\value{A \code{\link{list}} with one element per requested output of CIU results.
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
