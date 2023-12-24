import cv2
import numpy as np

def guided_filter(I, p, radius=15, eps=1e-3):
    # Ensure I and p are single-channel images
    if len(I.shape) == 3:
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    if len(p.shape) == 3:
        p = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)

    mean_I = cv2.boxFilter(I, -1, (radius, radius))
    mean_p = cv2.boxFilter(p, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(I * p, -1, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, -1, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    q = mean_a * I + mean_b
    threshold = 0.07
    # 你可以根据需要调整阈值

    # 二值化
    print(mean_a[499][450])
    binary_result = np.zeros_like(mean_a)
    binary_result[mean_a > threshold] = 255
    return binary_result

# Load an image
image = cv2.imread('./img./aurora_6.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
image = cv2.resize(image, (500, 500))
# Convert the image to grayscale
I = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define a guidance image (e.g., a smoothed version of the original image)
p = cv2.boxFilter(image, -1, (15, 15))

# Apply the guided filter
result = guided_filter(I, I, radius=2, eps=0.06)

# Display the original image and the result
cv2.imshow('Original Image', I)
cv2.imshow('Guided Filter Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
