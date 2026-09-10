def task8_birds_eye_view(img, corner_pts, out_size=400):
    """
    Given the 4 corners of chalk art that SHOULD form a perfect square,
    build a perspective transformation matrix that warps them into an
    actual square, producing a top-down "bird's-eye" view.

    We solve the 8-unknown homography by hand via the DLT approach (same
    method used in Task 9) rather than calling a built-in "compute
    homography" convenience function.
    """
    src = np.array(corner_pts, dtype=np.float64)  # 4 distorted corners (image)
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=np.float64)  # target perfect square

    H = compute_homography_DLT(src, dst)  # manual DLT (see Task 9 helper)
    result = cv2.warpPerspective(img, H, (out_size, out_size))
    return result, H
