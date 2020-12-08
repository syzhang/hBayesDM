#' @templateVar MODEL_FUNCTION generalise_gs
#' @templateVar CONTRIBUTOR \href{https://github.com/syzhang}{Suyi Zhang} <\email{suyizhang52@@gmail.com}>
#' @templateVar TASK_NAME Generalisation avoidance Task
#' @templateVar TASK_CODE generalise
#' @templateVar TASK_CITE 
#' @templateVar MODEL_NAME 6 Parameter Model
#' @templateVar MODEL_CODE gs
#' @templateVar MODEL_CITE (Norbury et al., 2018)
#' @templateVar MODEL_TYPE Hierarchical
#' @templateVar DATA_COLUMNS "subjID", "cue", "choice", "outcome"
#' @templateVar PARAMETERS \code{sigma_a} (shock generalisation), \code{sigma_n} (no loss generalisation), \code{eta} (Pearce Hall dynamic learning arbitration), \code{kappa} (Pearce Hall dynamic learning weight), \code{beta} (softmax temperature), \code{bias} (softmax bias)
#' @templateVar REGRESSORS 
#' @templateVar POSTPREDS "y_pred"
#' @templateVar LENGTH_DATA_COLUMNS 4
#' @templateVar DETAILS_DATA_1 \item{subjID}{A unique identifier for each subject in the data-set.}
#' @templateVar DETAILS_DATA_2 \item{cue}{Interger value representing the visual cue presented on the given trial, 1-7}
#' @templateVar DETAILS_DATA_3 \item{choice}{Integer value representing the avoidance choice on the given trial, 1 - avoided, 0 - not avoided}
#' @templateVar DETAILS_DATA_4 \item{outcome}{Floating point value representing the shock/loss outcome on the given trial (-1 or other losses).}
#' @templateVar LENGTH_ADDITIONAL_ARGS 0
#' 
#' @template model-documentation
#'
#' @export
#' @include hBayesDM_model.R
#' @include preprocess_funcs.R
#' 
#' @references
#' Norbury, Agnes, Trevor W Robbins, and Ben Seymour. (2018). ‘Value Generalization in Human Avoidance Learning’. Edited by Daeyeol Lee. ELife 7 (May): e34779. https://doi.org/10.7554/eLife.34779.
#'

generalise_gs <- hBayesDM_model(
  task_name       = "generalise",
  model_name      = "gs",
  model_type      = "",
  data_columns    = c("subjID", "cue", "choice", "outcome"),
  parameters      = list(
    "sigma_a" = c(0, 0.1, 1),
    "sigma_n" = c(0, 0.1, 1),
    "eta" = c(0, 0.1, 1),
    "kappa" = c(0, 0.1, 1),
    "beta" = c(0, 1, 10),
    "bias" = c(0, 0.1, 1)
  ),
  regressors      = NULL,
  postpreds       = c("y_pred"),
  preprocess_func = generalise_preprocess_func)
