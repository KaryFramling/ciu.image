#' Create `ciu.image.result` object.
#'
#' A `ciu.image.result` object is a [data.frame] with columns that correspond
#' to the passed parameters. There is one value/row per output.
#'
#' @param ci CI values.
#' @param cu CU values.
#' @param cmin Cmin values.
#' @param cmax Cmax values.
#' @param outval Output values.
#' @param out.names Output names.
#'
#' @return blabla
#' @export
ciu.image.result.new <- function(ci, cu, cmin, cmax, outval, out.names) {
  ciu.image.result <- data.frame(out.names=out.names, CI=ci, CU=as.numeric(cu),
                           cmin=cmin, cmax=cmax, outval=outval)
  class(ciu.image.result)<-c("ciu.image.result", "ciu.result", "data.frame")
  return(ciu.image.result)
}
