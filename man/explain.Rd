\name{explain}
\alias{explain}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Calculate CIU for specific instance and input/superpixel}
\description{Calculate Contextual Importance (CI) and Contextual Utility (CU)
for an image (Context) using the given "black-box" model.
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
explain(
  imgpath,
  ind.inputs.to.explain=c(1),
  in.min.max.limits=NULL,
  n.samples=100,
  n_superpixels=50,
  weight=20,
  n_iter=10,
  background = 'grey'
  target.concept=NULL,
  target.ciu=NULL
)
}
}

\arguments{
  \item{imgpath}{Path to image file to explain. }
  \item{ind.inputs.to.explain}{Indices of the input set to explain, i.e. one or
  more superpixel indices in this case.}
  \item{in.min.max.limits}{Not used.}
  \item{n.samples}{Not used.}
  \item{n_superpixels}{Number of superpixels to generate. Only done if superpixels
  don't exist yet for this `imgpath` or if the `imgpath` has changed from previous call. }
  \item{weight}{SLIC superpixel generation parameter.}
  \item{n_iter}{SLIC superpixel generation parameter.}
  \item{background}{Background colour to use. }
  \item{target.concept}{Not used yet.}
  \item{target.ciu}{Not used yet.}
}

\value{A \code{ciu.image.result} object as returned by
\code{\link{ciu.image.result.new}}, with rows sorted according to output value
from highest to lowest.
}
\seealso{
  \code{\link{ciu.image.result.new}}
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
