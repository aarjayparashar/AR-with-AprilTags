import numpy as np
import traceback

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    ##### STUDENT CODE END #####
    P = np.ones((3,3))
    P[0:2, 0] = Pc.T[:, 0]
    P[0:2, 1] = Pc.T[:, 1]
    P[0:2, 2] = Pc.T[:, 2]

    P_cal = np.linalg.inv(K)@P

    # f = (K[0][0] + K[1][1])/2 # focal length
    # M = np.ones(3,3)
    # M[0:2, 0] = Pc.T[0:2, 0]
    # M[0:2, 1] = Pc.T[0:2, 1]
    # M[0:2, 2] = Pc.T[0:2, 2]
    # M[-1, :] = np.array([f, f, f]).reshape(1, 3)

    # Unit vectors

    j1 = P_cal[:, 0]/np.linalg.norm(P_cal[:, 0])
    j2 = P_cal[:, 1]/np.linalg.norm(P_cal[:, 1])
    j3 = P_cal[:, 2]/np.linalg.norm(P_cal[:, 2])


    # j1 = (1/np.sqrt(np.square(Pc[0][0]) + np.square(Pc[0][1]) + np.square(f)))@M[:,0]
    # j2 = (1/np.sqrt(np.square(Pc[1][0]) + np.square(Pc[1][1]) + np.square(f)))@M[:,0]
    # j3 = (1/np.sqrt(np.square(Pc[2][0]) + np.square(Pc[2][1]) + np.square(f)))@M[:,0]

    # Cosine of angles
    cos_alpha = np.dot(j2, j3)
    cos_beta = np.dot(j1, j3)
    cos_gamma = np.dot(j1, j2)

    # distance between points in world frame

    p1 = Pw[0, :]
    p2 = Pw[1, :]
    p3 = Pw[2, :]

    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)

    a_b = a**2/b**2
    c_b = c**2/b**2

    A0 = (a_b - c_b - 1) ** 2 - (4 * c_b * cos_alpha * cos_alpha)

    A1 = 4 * (((a_b - c_b) * (1 - (a_b - c_b)) * cos_beta) - ((1 - (a_b + c_b)) * cos_alpha * cos_gamma) + (2 * c_b * cos_alpha * cos_alpha * cos_beta))

    A2 = 2 * ((a_b - c_b) ** 2 - 1 + (2 * (a_b - c_b) * (a_b - c_b) * cos_beta * cos_beta) + (2 * (1 - c_b) * cos_alpha * cos_alpha) - (4 * (a_b + c_b) * cos_alpha * cos_beta * cos_gamma) + (2 * (1 - a_b) * cos_gamma * cos_gamma))


    A3 = 4 * (-((a_b - c_b) * (1 + (a_b - c_b)) * cos_beta) + (2 * a_b * cos_gamma * cos_gamma * cos_beta) - ((1-(a_b + c_b)) * cos_alpha * cos_gamma))

    A4 = ((1 + a_b - c_b) ** 2) - (4 * a_b * cos_gamma * cos_gamma)

    coeff = [A0, A1, A2, A3, A4]
    v = np.roots(coeff)
    #v_real = v_1[np.real(v_1)].real
    v_real = []
    for root in v:
        if np.isreal(root) == True:
            v_real.append(np.real(root))

    err = []
    R_ = []
    t_ = []
    print(v_real)
    for v in v_real:

        u = (((-1 + a_b - c_b) * (v ** 2)) - (2 * (a_b - c_b) * cos_beta * v) + 1 + (a_b - c_b))/(2 * (cos_gamma - v * cos_alpha))
        s1_square = (c**2)/(1 + u**2 - 2*u*cos_gamma)
        s1 = np.sqrt(s1_square)
        s2 = u * s1
        s3 = v * s1


        # Pc_3d = np.vstack(s1/np.linalg.norm(P_cal[:, 0])) @ P_cal[:, 0]
        Pc_3d = np.vstack([s1*j1, s2*j2, s3*j3])



        # error = np.linalg.norm((K @ (R@ Pw.T[:, -1] + t))/((K @ (R @ Pw.T[:, -1] + t[-1]))[-1]) - Pc[-1, :])
        R,t = Procrustes(Pc_3d, Pw[0:3, :])
        R_.append(R)
        t_.append(t)

        # error = np.linalg.norm((K @ (R@ Pw.T[:, -1] + t))/((K @ (R @ Pw.T[:, -1] + t[-1]))[-1]) - Pc[-1, :])
        # temp = K @ R @ Pw[3, :].T
        temp = K@(R.T@ Pw[3, :].T - (R.T@t))
        temp = (temp/temp[2])[:2]
        error = np.linalg.norm(temp - Pc[3, :].T)
        err.append(error)

    index = np.argmin(err)
    print(index)

    R = R_[index]
    t = t_[index]



    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    A_mean = np.mean(Y, axis = 0)
    B_mean = np.mean(X, axis = 0)

    A = (Y - A_mean).T
    B = (X - B_mean).T

    C = A @ B.T

    [U, S, Vt] = np.linalg.svd(C)
    I = np.identity(3)
    I[2,2] = np.linalg.det(Vt.T@U.T)

    R = U @ I @ Vt



    t = A_mean - R@B_mean


##### STUDENT CODE END #####

    return R, t
