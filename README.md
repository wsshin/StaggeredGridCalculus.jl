# StaggeredGridCalculus

[![Build Status](https://travis-ci.org/wsshin/StaggeredGridCalculus.jl.svg?branch=master)](https://travis-ci.org/wsshin/StaggeredGridCalculus.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/dp8kcp896gghtbdx/branch/master?svg=true)](https://ci.appveyor.com/project/wsshin/staggeredgridcalculus-jl/branch/master)
[![codecov](https://codecov.io/gh/wsshin/MaxwellFDM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/wsshin/MaxwellFDM.jl)

StaggeredGridCalculus implements various differential and integral operators for vector
fields.  The vector fields are discretized on a staggered finite-difference grid (aka Yee's
grid) for 2nd-order accuracy in derivatives.

StaggeredGridCalculus is still under development, and it is intended to be used as an API
 for developing vector PDE solvers.
