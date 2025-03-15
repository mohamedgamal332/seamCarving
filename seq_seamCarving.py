import sys
import cv2 as cv
import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt

# Energy calculation using Sobel filters
def energy(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY).astype(np.float32)
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    return np.abs(sobel_x) + np.abs(sobel_y)

# Find the minimum seam using dynamic programming
def find_seam(energy_map):
    rows, cols = energy_map.shape
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int32)
    for i in range(1, rows):
        for j in range(cols):
            left = M[i-1, j-1] if j > 0 else float('inf')
            up = M[i-1, j]
            right = M[i-1, j+1] if j < cols-1 else float('inf')
            min_energy = min(left, up, right)
            backtrack[i, j] = j - 1 if min_energy == left else j + 1 if min_energy == right else j
            M[i, j] += min_energy
    return M, backtrack

# Remove a seam and highlight it
def carve_column(img):
    energy_map = energy(img)
    M, backtrack = find_seam(energy_map)
    mask = np.ones(img.shape[:2], dtype=np.bool_)
    seam_coords = []
    j = np.argmin(M[-1])
    for i in reversed(range(img.shape[0])):
        mask[i, j] = False
        seam_coords.append((i, j))
        j = backtrack[i, j]
    
    # Highlight removed seams on original image
    img_marked = img.copy()
    for i, j in seam_coords:
        img_marked[i, j] = [255, 0, 0]  # Red color for seam
    
    return img[mask].reshape((img.shape[0], img.shape[1] - 1, 3)), img_marked

# Seam carving process
def crop_c(img, scale_c):
    new_c = int(scale_c * img.shape[1])
    img_marked = img.copy()
    for _ in range(img.shape[1] - new_c):
        img, img_marked = carve_column(img)
    return img, img_marked

# Main function
def main(image_path, scale=0.9, output_path="output.jpg"):
    img = imread(image_path)
    resized_img, marked_img = crop_c(img, scale)
    imwrite(output_path, resized_img)
    imwrite("seam_highlighted.jpg", marked_img)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(marked_img)
    ax[0].set_title("Highlighted Seams")
    ax[0].axis("off")
    
    ax[1].imshow(resized_img)
    ax[1].set_title("Resized Image")
    ax[1].axis("off")
    
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python file.py <image_path> <scale>")
        sys.exit(1)
    main(sys.argv[1], float(sys.argv[2]))
