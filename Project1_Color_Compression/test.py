from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import time

flower = "phong_canh.jpg"

# The image is stored in a 3D array (height, width, RGB), containing RGB as integers from 0 to 255. 
# We can reshape this data to [n_samples x n_features], and rescale the color so they lie between 0 and 1.
data = flower / 255.0 
data = data.reshape(1000* 562, 3)

# Let's reduce the colors to just 20 using k-means clustering across the pixel space. 
# Due to the sixe of the dataset, we use mini batch k-means.
from sklearn.cluster import MiniBatchKMeans
start_time = time.time()
kmeans = MiniBatchKMeans(20)
kmeans.fit(data)
end_time = time.time()
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
#Let's see the results.
flower_compressed = new_colors.reshape(flower.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(flower)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(flower_compressed)
ax[1].set_title('20-color Image', size=16)

print("Run time: ", end_time-start_time)

plt.show() 