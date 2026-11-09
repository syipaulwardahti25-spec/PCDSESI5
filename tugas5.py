import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = 'lumba_lumba.jpg'  
img_bgr = cv2.imread(filename)

if img_bgr is None:
    raise FileNotFoundError(f"File {filename} tidak ditemukan. Pastikan file ada di folder kerja.")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

mean_3x3 = cv2.blur(img_rgb, (3, 3))
mean_5x5 = cv2.blur(img_rgb, (5, 5))

gauss_3x3 = cv2.GaussianBlur(img_rgb, (3, 3), 0)
gauss_5x5 = cv2.GaussianBlur(img_rgb, (5, 5), 0)

median_3x3 = cv2.medianBlur(img_rgb, 3)
median_5x5 = cv2.medianBlur(img_rgb, 5)

kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])
img_sharpen = cv2.filter2D(img_rgb, -1, kernel_sharpen)

titles = [
    '1. Asli', 
    '2. Mean 3x3', '3. Mean 5x5',
    '4. Gaussian 3x3', '5. Gaussian 5x5',
    '6. Median 3x3', '7. Median 5x5',
    '8. Sharpening'
]
images = [
    img_rgb, 
    mean_3x3, mean_5x5, 
    gauss_3x3, gauss_5x5, 
    median_3x3, median_5x5, 
    img_sharpen
]

plt.figure(figsize=(16, 8))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i], fontsize=11, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.savefig('hasil_pembahasan_filter.png', dpi=300)
plt.show()

print("=== Nilai Intensitas Piksel (Matriks Grayscale 5x5) ===")
print("\nCitra Asli:")
print(img_gray[:5, :5])

print("\nMean 3x3:")
print(cv2.blur(img_gray, (3, 3))[:5, :5])

print("\nGaussian 3x3:")
print(cv2.GaussianBlur(img_gray, (3, 3), 0)[:5, :5])

print("\nMedian 3x3:")
print(cv2.medianBlur(img_gray, 3)[:5, :5])

print("\nSharpening:")
print(cv2.filter2D(img_gray, -1, kernel_sharpen)[:5, :5])