import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

def dist_squared(point1, point2):
    assert point1.shape == point2.shape
    return np.sum(np.square(point2 - point1))

def find_closest_centroids(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def compute_centroids(X, labels, k_clusters):
    centroids = np.empty((k_clusters, X.shape[1]))
    for i in range(k_clusters):
        mask = (labels == i)
        if np.any(mask):
            centroids[i] = np.mean(X[mask], axis=0)
        else:
            centroids[i] = np.zeros(X.shape[1])
    return centroids

def run_kmeans(X, initial_centroids, k_clusters, max_iter):
    centroid_history = []
    centroids = initial_centroids
    for _ in range(max_iter):
        centroid_history.append(centroids)
        labels = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, labels, k_clusters)
    return labels, centroid_history

def choose_random_centroids(X, k_clusters):
    random_indices = np.random.choice(range(X.shape[0]), size=k_clusters, replace=False)
    return X[random_indices]

def image_compression(image_file, k_clusters, max_iter):
    # Đọc ảnh và chuyển thành mảng numpy
    image = Image.open(image_file)
    image_np = np.array(image)

    # Chuyển đổi ảnh thành 1D array
    img_1d = image_np.reshape(-1, 3) / 255.0  # Reshape và chuẩn hóa giá trị [0, 1]

    # Thực hiện K-Means
    initial_centroids = choose_random_centroids(img_1d, k_clusters)
    labels, centroid_history = run_kmeans(img_1d, initial_centroids, k_clusters, max_iter)

    # Tạo ảnh mới từ các centroid và nhãn
    final_centroids = centroid_history[-1]
    final_image = final_centroids[labels].reshape(image_np.shape)

    # Hiển thị ảnh gốc và ảnh sau khi giảm số lượng màu
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("Original Image")
    axs[0].imshow(image_np)
    axs[0].axis('off')
    axs[1].set_title(f"Reduced Color Image (k={k_clusters})")
    axs[1].imshow(final_image)
    axs[1].axis('off')
    plt.show()

    # Lưu ảnh đầu ra
    output_file = "output_image.png"
    Image.fromarray((final_image * 255).astype(np.uint8)).save(output_file)
    print("Lưu ảnh đầu ra thành công!")

# Chạy chương trình
image_file = input("Nhập tên tập tin ảnh: ")
k_clusters = int(input("Nhập số lượng màu cần giảm: "))
max_iter = int(input("Nhập số lần lặp tối đa: "))

start_time = time.time()
image_compression(image_file, k_clusters, max_iter)
end_time = time.time()

elapsed_time = end_time - start_time
print ("Thời gian ra kết quả:{0}".format(elapsed_time) + "[sec]")
