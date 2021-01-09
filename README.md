# StaggeredGridCalculus

[![Build Status](https://travis-ci.com/wsshin/StaggeredGridCalculus.jl.svg?branch=main)](https://travis-ci.com/wsshin/StaggeredGridCalculus.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/dp8kcp896gghtbdx/branch/main?svg=true)](https://ci.appveyor.com/project/wsshin/staggeredgridcalculus-jl/branch/main)
[![codecov](https://codecov.io/gh/wsshin/StaggeredGridCalculus.jl/branch/main/graph/badge.svg?token=SONTDLL9GY)](https://codecov.io/gh/wsshin/StaggeredGridCalculus.jl)

StaggeredGridCalculus is a Julia package that implements various differential and integral
operators for vector fields.  The vector fields are discretized on a staggered
finite-difference grid (aka Yee's grid) for 2nd-order accuracy in the result.

StaggeredGridCalculus is still under development, and it is intended to be used as an API
 for developing vector PDE solvers.
