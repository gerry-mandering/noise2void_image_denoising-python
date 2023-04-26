import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread  # read Image
from skimage.transform import resize  # resize Image
from sklearn.neighbors import KNeighborsClassifier
from keras.preprocessing.image import ImageDataGenerator

# 1. 이미지 읽어 들이기
url = 'https://github.com/dknife/ML/raw/main/data/Proj1/40/'

imgRow, imgCol, channel = 24, 24, 3
images = []

for i in range(40):
    file = url + 'img{0:02d}.jpg'.format(i + 1)  # url + 이미지 주소 하나씩 가져오기
    img = imread(file)  # imread 사용해서 이미지 읽기
    img = resize(img, (imgRow, imgCol, channel))  # resize 사용해서 이미지 크기 24X24로 맞추고 색상 채널 RGB로 설정
    images.append(img)  # images 배열에 해당 이미지 append


def plot_images(nRow, nCol, img):
    fig = plt.figure()
    fig, ax = plt.subplots(nRow, nCol, figsize=(nCol, nRow))
    for i in range(nRow):
        for j in range(nCol):
            if nRow <= 1:
                axis = ax[j]
            else:
                axis = ax[i, j]
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            axis.imshow(img[i * nCol + j])

# plot_images(4, 10, images)

# 2. 훈련용 데이터와 검증용 데이터 분리
X = np.array(images[:30])
X_test = np.array(images[30:])
# plot_images(3, 10, X)
# plot_images(1, 10, X_test)

# 3. 입력 데이터 준비
X_noisy = X + np.random.randn(len(X), imgRow, imgCol, channel) * 0.1
X_noisy = np.clip(X_noisy, 0, 1)
X_test_noisy = X_test + np.random.randn(len(X_test), imgRow, imgCol, channel) * 0.1
X_test_noisy = np.clip(X_test_noisy, 0, 1)

# plot_images(3, 10, X_noisy)
# plot_images(1, 10, X_test_noisy)
# plt.savefig('X_test_noisy.png')

# 4. 분류기 입출력 데이터 형식에 맞추어 훈련하기
X_noisy_flat = X_noisy.reshape(-1, imgRow * imgCol * channel)
X_flat = np.array(X.reshape(-1, imgRow * imgCol * channel) * 255, dtype=np.uint)

knn = KNeighborsClassifier()
knn.fit(X_noisy_flat, X_flat)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=-1, n_neighbors=5,
                     p=2, weights='uniform')

denoised_result = knn.predict(X_noisy_flat)
denoised_result = denoised_result.reshape(-1, imgRow, imgCol, channel)
# plot_images(3, 10, denoised_result)

# 5. 데이터를 증강하여 훈련 효과 높이기
n_augmentation = 100
X_noisy_aug = X + np.random.rand(len(X), imgRow, imgCol, channel) * 0.2
y_label = np.array(X * 255, dtype=np.uint)
y = y_label

print(X_noisy.shape)
for i in range(n_augmentation):
    noisy_data = X + np.random.randn(len(X), imgRow, imgCol, channel) * 0.2
    X_noisy_aug = np.append(X_noisy_aug, noisy_data, axis=0)
    y = np.append(y, y_label, axis=0)

X_noisy_aug = np.clip(X_noisy_aug, 0, 1)
print(X_noisy_aug.shape, y.shape)

# plot_images(1, 10, X_noisy_aug[0:300:30])

X_noisy_aug_flat = X_noisy_aug.reshape(-1, imgRow * imgCol * channel)
y_flat = y.reshape(-1, imgRow * imgCol * channel)

knn.fit(X_noisy_aug_flat, y_flat)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=-1, n_neighbors=5,
                     p=2, weights='uniform')

denoised_result = knn.predict(X_noisy_flat)
denoised_result = denoised_result.reshape(-1, imgRow, imgCol, channel)

# plot_images(3, 10, X_noisy)
# plot_images(3, 10, denoised_result)

# 6. 검증 데이터로 일반화 능력을 살펴보자
rndidx = np.random.randint(0, 20)
data = X[rndidx:rndidx + 10] + np.random.randn(10, imgRow, imgCol, channel) * 0.4
data = np.clip(data, 0, 1)
data_flat = data.reshape(-1, imgRow * imgCol * channel)

denoised = knn.predict(data_flat)
denoised = denoised.reshape(-1, imgRow, imgCol, channel)
denoised = np.clip(denoised, 0, 255)

# plot_images(1, 10, data)
# plot_images(1, 10, denoised)

denoised = knn.predict(X_test_noisy.reshape(-1, imgRow * imgCol * channel))
denoised = denoised.reshape(-1, imgRow, imgCol, channel)

# plot_images(1, 10, X_test_noisy)
# plot_images(1, 10, denoised)

# 7. 데이터 증강으로 일반화 능력을 높여보자
image_generator = ImageDataGenerator(
    rotation_range=360,
    zoom_range=0.1,
    shear_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)

y_aug = X.reshape(-1, imgRow, imgCol, channel)
it = image_generator.flow(y_aug)
nData = y_aug.shape[0]

X_aug = y_aug + np.random.randn(nData, imgRow, imgCol, channel) * 0.1

n_augmentation = 500
for _ in range(n_augmentation):
    new_y = it.next()
    new_X = new_y + np.random.randn(nData, imgRow, imgCol, channel) * 0.1
    y_aug = np.append(y_aug, new_y)
    X_aug = np.append(X_aug, new_X)

y_aug = np.array(y_aug * 255, dtype=np.uint)
y_aug = y_aug.reshape(-1, imgRow, imgCol, channel)
X_aug = X_aug.reshape(-1, imgRow, imgCol, channel)

y_aug = np.clip(y_aug, 0, 255)
X_aug = np.clip(X_aug, 0, 1)
plot_images(3, 10, y_aug[30:])
plot_images(3, 10, X_aug[30:])

# 8. 새로 학습하고 검증용 데이터 적용하기
X_aug_flat = X_aug.reshape(-1, imgRow * imgCol * channel)
y_aug_flat = y_aug.reshape(-1, imgRow * imgCol * channel)
knn.fit(X_aug_flat, y_aug_flat)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=-1, n_neighbors=5,
                     p=2, weights='uniform')

denoised = knn.predict(X_test_noisy.reshape(-1, imgRow * imgCol * channel))
denoised = denoised.reshape(-1, imgRow, imgCol, channel)
plot_images(1, 10, X_test_noisy)
plt.savefig('X_test_noisy_aug.png')
plot_images(1, 10, denoised)
plt.savefig('denoised_aug.png')
plt.show()
#
# images = it.next()
# testX = images + np.random.randn(nData, imgRow, imgCol, channel) * 0.4
# testX = np.clip(testX, 0, 1)
# denoised = knn.predict(testX.reshape(-1, imgRow * imgCol * channel))
# denoised = denoised.reshape(-1, imgRow, imgCol, channel)
#
# # plot_images(1, 10, testX)
# # plot_images(1, 10, denoised)
# # plt.show()
