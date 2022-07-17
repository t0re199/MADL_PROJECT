import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from image.ImagePreProcessing import labels_to_int, normalize_dataset
from spio.ImageLoading import load_image_dataset
from spio.Models import save_object, load_object, exists_checkpoint

CHECKPOINT_FILE = "rgb_dataset_pca.svd"


def load_rgb_pca_dataset(normalize=True):
    dataset, labels = load_image_dataset(flatten=True)

    mean = np.mean(dataset, axis=0)
    dataset = dataset - mean

    if normalize:
        dataset = normalize_dataset(dataset)
    labels = labels_to_int(labels)

    if exists_checkpoint(CHECKPOINT_FILE):
        dataset = load_object(CHECKPOINT_FILE)
    else:
        pca = PCA(n_components=0x1e)
        dataset = pca.fit_transform(dataset)
        save_object(dataset, CHECKPOINT_FILE)

    return dataset, labels


def ds2_dump(data, datapath):
    import struct

    cols = 0x1 if len(data.shape) == 0x1 else data.shape[0x1]
    rows = data.shape[0x0]
    with open(datapath, "wb") as fd:
        fd.write(struct.pack("<i", cols))
        fd.write(struct.pack("<i", rows))
        for element in data.flatten():
            fd.write(element)


def ds2_load(datapath, dtype):
    with open(datapath, "rb") as fd:
        cols = np.frombuffer(fd.read(0x4), dtype=np.int32)[0x0]
        rows = np.frombuffer(fd.read(0x4), dtype=np.int32)[0x0]

        data = np.zeros((rows, cols), dtype=dtype)

        size = 0x4 if dtype == np.float32 else 0x8
        for i in range(rows):
            data[i] = np.frombuffer(fd.read(size * cols), dtype=dtype, count=cols)
        return data


def plot_class_hist(labels):

    labels_set = np.unique(labels)

    items_dict = dict()
    for label in labels_set:
        amount = (labels == label).sum()
        items_dict[label] = amount

    print(items_dict)
    colors = ["cyan", "lighblue", "steelblue", "cornflowerblue"]
    plt.set_cmap("Blues")
    bars = plt.bar([str(int(k)) for k in items_dict.keys()], height=items_dict.values())
    plt.yticks(np.arange(0, max(items_dict.values()) + 100, 100))
    plt.show()

if __name__ == '__main__':
    np.random.seed(0xd)
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_confusion_matrix

    n_comp = 1000
    dtype = np.float32
    dtype_str = "sp" if dtype == np.float32 else "dp"

    dataset, labels = load_image_dataset(flatten=True)
    dataset = dataset.astype(dtype)

    labels = labels_to_int(labels)
    labels = labels.astype(dtype)

    mean = np.mean(dataset, axis=0)
    dataset = dataset - mean

    dataset = normalize_dataset(dataset)

    indexes_0 = np.argwhere(labels == 0x0).reshape(-1)
    indexes_2 = np.argwhere(labels == 0x2).reshape(-1)

    arch_dataset_10 = np.concatenate([dataset[indexes_0], dataset[indexes_2]])
    arch_labels = np.ones((arch_dataset_10.shape[0x0], 0x1), dtype=dtype)
    arch_labels[indexes_0.shape[0x0]:] *= -0x1

    permutation = np.random.permutation(arch_dataset_10.shape[0x0])

    arch_dataset_10 = arch_dataset_10[permutation]
    arch_labels = arch_labels[permutation].reshape(-0x1)

    pca = PCA(n_components=n_comp)
    arch_dataset_10 = pca.fit_transform(arch_dataset_10)

    X_train, X_test, y_train, y_test = train_test_split(arch_dataset_10,
                                                        arch_labels,
                                                        test_size=0.30,
                                                        random_state=0xd)

    ds2_dump(X_test, f"/Users/t0re199/git/archp_project/data/arch_1k_{n_comp}_{dtype_str}.data")
    ds2_dump(y_test, f"/Users/t0re199/git/archp_project/data/arch_1k_{n_comp}_{dtype_str}.labels")

    support_vector_classifier = SVC(decision_function_shape="ovo",
                                    C=0x64,
                                    gamma=0.001,
                                    kernel="poly",
                                    degree=0x2)
    support_vector_classifier.fit(arch_dataset_10, arch_labels.reshape(-1))
    y_pred = support_vector_classifier.predict(arch_dataset_10)

    disp = plot_confusion_matrix(support_vector_classifier, arch_dataset_10, arch_labels,
                                 display_labels=np.unique(arch_labels),
                                 cmap="Blues",
                                 normalize="true")
    disp.ax_.set_title("PCA Support Vector Machine Confusion Matrix RGB")
    plt.show()
