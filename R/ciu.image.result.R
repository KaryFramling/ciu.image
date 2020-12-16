#' ciu.image.result.new
#'
#' @param ci blabla
#' @param cu blabla
#' @param cmin blabla
#' @param cmax blabla
#' @param outval blabla
#' @param out.names blabla
#'
#' @return blabla
#' @export
ciu.image.result.new <- function(ci, cu, cmin, cmax, outval, out.names) {
  ciu.image.result <- data.frame(out.names=out.names, CI=ci, CU=as.numeric(cu),
                           cmin=cmin, cmax=cmax, outval=outval)
  class(ciu.image.result)<-c("ciu.image.result", "ciu.result", "data.frame")
  return(ciu.image.result)
}
