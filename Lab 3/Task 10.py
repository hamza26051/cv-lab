def task10_insert_painting(painting_img, frame_corners, canvas_size):
    """
    Traverse the FULL transformation hierarchy to insert a flat painting
    into an angled picture frame:
        1. Linear Scale     -> shrink the painting
        2. Rigid Transform  -> move it near the frame (rotation + translation)
        3. Projective Warp  -> map its 4 corners exactly onto the angled
                                frame's 4 corners in the final scene
    """
    h, w = painting_img.shape[:2]

    # --- Step 1: Linear Scale (shrink painting) ---
    scale = 0.5
    Scale2x2 = np.array([
        [scale, 0.0],
        [0.0,   scale]
    ], dtype=np.float64)
    new_w, new_h = int(w * scale), int(h * scale)
    M_scale = np.array([
        [Scale2x2[0, 0], Scale2x2[0, 1], 0],
        [Scale2x2[1, 0], Scale2x2[1, 1], 0]
    ], dtype=np.float64)
    scaled = cv2.warpAffine(painting_img, M_scale, (new_w, new_h))

    # --- Step 2: Rigid Transform (rotate slightly + move near frame) ---
    angle_deg = 5.0
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    tx, ty = 20, 15  # nudge toward the frame's approximate position
    Rigid = np.array([
        [cos_t, -sin_t, tx],
        [sin_t,  cos_t, ty],
        [0.0,    0.0,   1.0]
    ], dtype=np.float64)
    M_rigid = Rigid[:2, :]
    positioned = cv2.warpAffine(scaled, M_rigid, (new_w + 60, new_h + 60))

    # --- Step 3: Projective Transform (warp 4 corners onto angled frame) ---
    ph, pw = positioned.shape[:2]
    src_corners = np.array([
        [0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]
    ], dtype=np.float64)
    dst_corners = np.array(frame_corners, dtype=np.float64)  # angled frame's 4 corners

    H = compute_homography_DLT(src_corners, dst_corners)
    warped_painting = cv2.warpPerspective(positioned, H, canvas_size)

    # Build a mask of the warped painting so we can composite it onto the wall
    mask = cv2.warpPerspective(
        np.ones((ph, pw), dtype=np.uint8) * 255, H, canvas_size
    )
    return warped_painting, mask, (Scale2x2, Rigid, H)
    # ---------------------------------------------------------------------------
# Demo / self-test harness
# ---------------------------------------------------------------------------
def make_test_image(w=200, h=150, label="TEST"):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (0, 200, 255), 3)
    cv2.putText(img, label, (15, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    # a few landmark dots for tasks that need matching points
    for (x, y) in [(20, 20), (w - 20, 20), (w - 20, h - 20), (20, h - 20)]:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    return img


if __name__ == "__main__":
    base = make_test_image()
    save("00_original.png", base)

    r1, S, M1 = task1_scale_fingerprint(base)
    save("task1_scaled.png", r1)

    r2, R, M2 = task2_derotate_satellite(base, angle_deg=45)
    save("task2_derotated.png", r2)

    r3, Sh, M3 = task3_deshear_barcode(base, shear_factor=-0.5)
    save("task3_desheared.png", r3)

    r4, T4 = task4_translate_map(base, shift_left=150, shift_up=80)
    save("task4_translated.png", r4)

    r5, Rigid5 = task5_rigid_align(base, angle_deg=30, tx=40, ty=-25)
    save("task5_rigid.png", r5)

    r6, Sim6 = task6_similarity_fix(base, scale=1.8, angle_deg=-20, tx=60, ty=30)
    save("task6_similarity.png", r6)

    h, w = base.shape[:2]
    src7 = [[20, 20], [w - 20, 20], [20, h - 20]]           # clean landmarks
    dst7 = [[35, 40], [w - 10, 30], [50, h - 15]]           # glitched landmarks
    r7, M7 = task7_fix_affine_glitch(base, src7, dst7)
    save("task7_deglitched.png", r7)

    corners8 = [[30, 40], [w - 15, 10], [w - 30, h - 10], [15, h - 30]]
    r8, H8 = task8_birds_eye_view(base, corners8, out_size=300)
    save("task8_birdseye.png", r8)

    base2 = make_test_image(label="CAM2")
    pts_img1 = [[w - 60, 20], [w - 10, 20], [w - 10, h - 20], [w - 60, h - 20]]
    pts_img2 = [[10, 20], [60, 20], [60, h - 20], [10, h - 20]]
    r9, H9 = task9_stadium_panorama(base, base2, pts_img1, pts_img2)
    save("task9_panorama.png", r9)

    painting = make_test_image(w=150, h=100, label="ART")
    canvas = np.full((400, 400, 3), 60, dtype=np.uint8)
    frame_corners = [[80, 60], [320, 30], [340, 300], [60, 320]]  # angled frame
    warped, mask, mats = task10_insert_painting(painting, frame_corners, (400, 400))
    composite = canvas.copy()
    mask_bool = mask > 0
    composite[mask_bool] = warped[mask_bool]
    save("task10_final_composite.png", composite)

    print("\nAll 10 tasks executed successfully. Check ./outputs/")
