import numpy as np

def est_homography(X, Y):
    """
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out
    what X and Y should be.
    Input:
        X: 4x2 matrix of (x,y) coordinates
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    A = []

    for i in range(X.shape[0]):
        a_x = np.array([-X[i][0], -X[i][1], -1, 0, 0, 0, np.multiply(X[i][0], Y[i][0]), np.multiply(X[i][1], Y[i][0]), Y[i][0]])
        a_y = np.array([0, 0, 0, -X[i][0], -X[i][1], -1, np.multiply(X[i][0], Y[i][1]), np.multiply(X[i][1], Y[i][1]), Y[i][1]])
        A.append(a_x)
        A.append(a_y)

    [U, S, Vt] = np.linalg.svd(np.array(A))
    V = np.transpose(Vt)
    H = V[:, -1].reshape(3,3)
    #print(H)

    ##### STUDENT CODE END #####
    return H
