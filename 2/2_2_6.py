from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("slice ------")

my_slice = train_images[10:100]
print(my_slice.shape)

my_slice = train_images[10:100, :, :]
print(my_slice.shape)

my_slice = train_images[10:20, 5:20, 5:20]
print(my_slice.shape)

print("batch ------")

batch = train_images[:128]
print(batch.shape)

batch = train_images[128:256]
print(batch.shape)

n = 5
batch = train_images[128 * n:128 * (n + 1)]
print(batch.shape)
