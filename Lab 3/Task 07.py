def task7_fix_affine_glitch(img, src_pts, dst_pts):
    """
    Given 3 matching landmark points (before/after the glitch), solve a
    system of linear equations for the 6 unknowns of a general 2x3 affine
    matrix:
        [x']   [a  b  tx] [x]
        [y'] = [c  d  ty] [y]
                          [1]
    We are given dst_pts = A @ src_pts_h, and want the matrix A that
    UNDOES the glitch, i.e. maps the glitched (dst) points back to the
    clean (src) points.
    """
    src_pts = np.array(src_pts, dtype=np.float64)  # 3 clean landmark points
    dst_pts = np.array(dst_pts, dtype=np.float64)  # 3 glitched landmark points

    # We solve for the affine matrix A_fix such that:
    #     src = A_fix @ [dst; 1]
    # Build the 6x6 linear system A_sys * p = b, where p = [a,b,tx,c,d,ty]
    A_sys = np.zeros((6, 6), dtype=np.float64)
    b = np.zeros((6,), dtype=np.float64)

    for i in range(3):
        x, y = dst_pts[i]
        xp, yp = src_pts[i]
        # row for x' = a*x + b*y + tx
        A_sys[2 * i]     = [x, y, 1, 0, 0, 0]
        b[2 * i]         = xp
        # row for y' = c*x + d*y + ty
        A_sys[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i + 1]     = yp

    p = np.linalg.solve(A_sys, b)
    a, b_, tx, c, d, ty = p

    M = np.array([
        [a, b_, tx],
        [c, d,  ty]
    ], dtype=np.float64)

    h, w = img.shape[:2]
    result = cv2.warpAffine(img, M, (w, h))
    return result, M
