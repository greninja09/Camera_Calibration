import cv2
import numpy as np
import glob

# 체스보드 사이즈
chessboard_size = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D world points (Z=0)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('./calibration_images/*.jpg')  # 추출한 체스보드 이미지들

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

cv2.destroyAllWindows()

# 카메라 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print(f'''## Camera Calibration Results
* The number of applied images = {len(images)}
* RMS error = {mean_error/len(objpoints)}
* Camera matrix K =
{mtx}
* Distortion Coefficient (k1, k2, p1, p2, k3, ...) =
{dist}
''')


# 이미지에서 왜곡 제거 (ChatGPT도움)
for fname in images:
    iname = fname.split('\\')[1]
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # 왜곡 보정
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # 잘라내기 (ROI)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    cv2.imwrite(f'./distortion_images/{iname}', dst)
