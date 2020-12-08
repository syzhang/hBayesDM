context("Test generalise_gs")
library(hBayesDM)

test_that("Test generalise_gs", {
  # Do not run this test on CRAN
  skip_on_cran()

  expect_output(generalise_gs(
      data = "example", niter = 10, nwarmup = 5, nchain = 1, ncore = 1))
})
