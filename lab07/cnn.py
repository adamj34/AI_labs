# import numpy as np
# from matplotlib import pyplot as plt
# from scipy.signal import convolve

# # tworzymy tablice o wymiarach 128x128x3 (3 kanaly to RGB)
# # uzupelnioną zerami = kolor czarny
# data = np.zeros((128, 128, 3), dtype=np.uint8)


# # chcemy zeby obrazek byl czarnobialy,
# # wiec wszystkie trzy kanaly rgb uzupelniamy tymi samymi liczbami
# # napiszmy do tego funkcje
# def draw(img, x, y, color):
#     img[x, y] = [color, color, color]


# # zamalowanie 4 pikseli w lewym górnym rogu
# draw(data, 5, 5, 100)
# draw(data, 6, 6, 100)
# draw(data, 5, 6, 255)
# draw(data, 6, 5, 255)


# # rysowanie kilku figur na obrazku
# for i in range(128):
#     for j in range(128):
#         if (i-64)**2 + (j-64)**2 < 900:
#             draw(data, i, j, 200)
#         elif i > 100 and j > 100:
#             draw(data, i, j, 255)
#         elif (i-15)**2 + (j-110)**2 < 25:
#             draw(data, i, j, 150)
#         elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
#             draw(data, i, j, 255)

# filter_horizontal = np.array([[1, 0, -1],
#                              [1, 0, -1],
#                              [1, 0, -1]]) 

# filter_vertical = np.array([[1, 1, 1],
#                             [0, 0, 0],
#                             [-1,-1,-1]])

# filter_oblique = np.array([[0, 1, 2],
#                           [-1, 0, 1],
#                           [-2,-1, 0]])

# def con(arr, kernel):
#     result = np.zeros((126, 126), dtype=np.float32)
#     for i in range(3):
#         result += convolve(arr[:,:,i], kernel, mode='valid')
#     return result

# # konwersja macierzy na obrazek i wyświetlenie
# plt.imshow(r, interpolation='nearest', cmap='gray')
# plt.show()

import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

def draw(img, x, y, color):
    img[x, y] = [color, color, color]

data = np.zeros((128, 128, 3), dtype=np.uint8)

draw(data, 5, 5, 100)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)

for i in range(128):
    for j in range(128):
        if (i-64)**2 + (j-64)**2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i-15)**2 + (j-110)**2 < 25:
            draw(data, i, j, 150)
        elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
            draw(data, i, j, 255)


# define kernels
horizontal_kernel = np.array([[1, 0, -1],
                              [1, 0, -1],
                              [1, 0, -1]]) 

vertical_kernel = np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1,-1,-1]])

sobel_kernel = np.array([[0, 1, 2],
                        [-1, 0, 1],
                        [-2,-1, 0]])

data = data[..., 0]
print(data)

# apply kernels with stride=1 and no padding
vertical_edges = convolve2d(data[1:-1], vertical_kernel[::-1], mode='valid')
horizontal_edges = convolve2d(data[1:-1], horizontal_kernel[::-1], mode='valid')
sobel_edges = convolve2d(data[1:-1], sobel_kernel[::-1], mode='valid')

# apply kernels with stride=2 and no padding
vertical_edges_s2 = convolve2d(data[1:-1, 1:-1], vertical_kernel[::-1], mode='valid')[::2, ::2]
horizontal_edges_s2 = convolve2d(data[1:-1, 1:-1], horizontal_kernel[::-1], mode='valid')[::2, ::2]
sobel_edges_s2 = convolve2d(data[1:-1, 1:-1], sobel_kernel[::-1], mode='valid')[::2, ::2]

# plot results
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(331)
ax.imshow(vertical_edges,cmap='gray')
ax.set_title('Vertical Edges Stride=1')
ax.axis('off')

ax = fig.add_subplot(332)
ax.imshow(horizontal_edges,cmap='gray')
ax.set_title('Horizontal Edges Stride=1')
ax.axis('off')

ax = fig.add_subplot(333)
ax.imshow(sobel_edges,cmap='gray')
ax.set_title('Sobel Edges Stride=1')
ax.axis('off')

ax = fig.add_subplot(334)
ax.imshow(vertical_edges_s2,cmap='gray')
ax.set_title('Vertical Edges Stride=2')
ax.axis('off')

ax = fig.add_subplot(335)
ax.imshow(horizontal_edges_s2,cmap='gray')
ax.set_title('Horizontal Edges Stride=2')
ax.axis('off')

ax = fig.add_subplot(336)
ax.imshow(sobel_edges_s2,cmap='gray')
ax.set_title('Sobel Edges Stride=2')
ax.axis('off')

plt.show()
