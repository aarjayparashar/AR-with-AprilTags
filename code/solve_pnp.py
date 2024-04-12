from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    # Pw = np.delete(Pw, 2, axis = 1)
    H = est_homography(Pw, Pc) # estimating homography matrix, H
    H = H/H[2,2] # normalizing H
    H_prime = np.linalg.inv(K)@H # K_inverse * H

    h1 = H_prime[:, 0]
    h2 = H_prime[:, 1]
    h3 = H_prime[:, 2]
    # h3 = np.cross(h1, h2)

    # R = np.zeros([3, 3])
    # R[:, 0] = h1
    # R[:, 1] = h2
    # R[:, 2] = h3

    # R = np.array([h1, h2, np.cross(h1, h2)]).T
    R = np.hstack((h1.reshape(3,1), h2.reshape(3,1), np.cross(h1, h2).reshape(3,1)))
    # print(R)
    [U, S, Vt] = np.linalg.svd(R)
    # I = np.eye(3,3)
    I = np.identity(3)
    I[2,2] = np.linalg.det(U@Vt)

    R = U @ I @ Vt
    # t = H_prime[:, 2]/np.linalg.norm(h1)
    t = h3/np.linalg.norm(h1)

    # R = np.linalg.inv(R)
    # t = np.matmul(-R, t)

    R = np.linalg.inv(R)
    t = -R@t

    return R, t
