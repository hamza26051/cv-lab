def task9_stadium_panorama(img1, img2, pts_img1, pts_img2):
    """
    Given 4 matching points found in both camera images, compute the
    Homography matrix that projects the second camera's view into the
    first camera's coordinate space, then stitch them into one panorama.
    """
def compute_homography_DLT(src_pts, dst_pts):
    """
    Manually compute the Homography matrix H (3x3, up to scale) that maps
    src_pts -> dst_pts using the Direct Linear Transform (DLT) with (at
    least) 4 point correspondences. This is the classic "solve via SVD"
    method: build the 2n x 9 matrix A, take the singular vector for the
    smallest singular value, reshape to 3x3.
    """
    src_pts = np.array(src_pts, dtype=np.float64)
    dst_pts = np.array(dst_pts, dtype=np.float64)
    n = src_pts.shape[0]
    assert n >= 4, "Homography needs at least 4 point correspondences"

    A = []
    for i in range(n):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A, dtype=np.float64)

    # Solve via SVD: h is the eigenvector for the smallest singular value
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    H = H / H[2, 2]  # normalize so H[2,2] = 1
    return H
