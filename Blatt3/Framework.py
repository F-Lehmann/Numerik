import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import pyplot


def ComputeTPSWeights(X, Y, Z):
    """This function constructs a thin plate spline interpolating the given three-
       dimensional points by means of a two-dimensional function. 
      \param X,Y,Z Three arrays of shape (n,) containing n points in three-
             dimensional space. 
      \return An array of shape identical to X where the i-th entry stores the 
              weight to be used for the radial basis function centered at 
              (X[i],Y[i])."""
    m = X.shape[0]
    A = np.zeros(shape=(m, m))
    for i in range(m):
        vi = np.array([X[i], Y[i]])
        for j in range(m):
            vj = np.array([X[j], Y[j]])
            r = np.linalg.norm(vi-vj)
            A[j, i] = (r**2) * np.log(max(1.0e-8,r))
    w = np.linalg.solve(A, Z)
    return w


def EvaluateTPSSpline(XNew, YNew, X, Y, Weights):
    """Given the weights for a thin plate spline as returned by ComputeTPSWeights 
       this function evaluates the thin plate spline at prescribed locations.
      \param XNew,YNew The x and y coordinates at which the TPS spline should be 
             evaluated. These are np.ndarray objects of arbitrary but identical 
             shape.
      \param X,Y The coordinates passed to ComputeTPSWeights().
      \param Weights The weights returned by ComputeTPSWeights().
      \return An array of shape identical to XNew containing the value of the thin 
              plate spline at the coordinates given by XNew and YNew."""
    if (len(XNew.shape) > 1):
        XNew = XNew[0, :]
    if (len(YNew.shape) > 1):
        YNew = YNew[:, 0]
    m = XNew.shape[0]
    res = np.zeros(shape=(m, m))
    for ix in range(m):
        for iy in range(m):
            tmp = 0
            for j in range(len(Weights)):
                r = np.linalg.norm(np.array([XNew[ix]-X[j],YNew[iy]-Y[j]]))
                tmp += Weights[j] * ((r**2) * np.log(max(1.0e-8,r)))
            res[iy, ix] = tmp
    return res

if(__name__ == "__main__"):
    # Produce random points which are to be interpolated by the thin plate spline
    nPoint = 20
    X = np.random.rand(nPoint)
    Y = np.random.rand(nPoint)
    Z = np.random.rand(nPoint)
    # Produce a regular grid for evaluation of the thin plate spline
    nGridCell = 41
    XNew = np.linspace(0.0, 1.0, nGridCell)
    YNew = np.linspace(0.0, 1.0, nGridCell)
    XNew, YNew = np.meshgrid(XNew, YNew)
    # Construct and evaluate the thin plate spline
    Weights = ComputeTPSWeights(X, Y, Z)
    ZNew = EvaluateTPSSpline(XNew, YNew, X, Y, Weights)

    # Check whether all points have been fitted correctly
    ZReconstructed = EvaluateTPSSpline(X, Y, X, Y, Weights)
    print("If the following number is nearly zero, the solution appears to be working fine.")
    print(np.linalg.norm(Z-ZReconstructed))

    # Plot the input points and the interpolated function
    Axes = pyplot.subplot(projection="3d")
    Axes.scatter3D(X, Y, Z, color="r")
    Axes.set_xlim(0.0, 1.0)
    Axes.set_ylim(0.0, 1.0)
    Axes.set_zlim(np.min(ZNew), np.max(ZNew))
    Axes.plot_wireframe(XNew, YNew, ZNew, rstride=1, cstride=1)
    pyplot.show()
