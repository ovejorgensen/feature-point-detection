import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

TEST_NUM = 100

bf_times = []
flann_times = []

images =  ['hus', 'flaske', 'smoreost', 'kopp']

for image in images:
    img1 = cv2.imread(f'testing_images/{image}.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f'testing_images/{image}2.jpg', cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(f'{image}:')

    bf_start = time.time()
    for _ in tqdm(range(TEST_NUM)):
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        bf_matches = bf.match(des1,des2)
    t = time.time() - bf_start
    # bf_times.append((t, len(bf_matches)))
    bf_times.append(t)


    flann_start = time.time()
    for _ in tqdm(range(TEST_NUM)):
        flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=3), dict(checks=50))
        flann_matches = flann.match(des1,des2)
    t = time.time() - flann_start
    # flann_times.append((t, len(flann_matches)))
    flann_times.append(t)

plt.plot(bf_times, label='BFMatcher')
plt.plot(flann_times, label='FlannBasedMatcher')
plt.xticks(range(len(images)), images)
plt.legend()
plt.show()