import os
import numpy as numpy
from skimage import io, transform
import skimage
import numpy as np

weight = 100
hight = 100
data = None
labels = None

domains = ["train", "validation"]


def process_image(image, weight, hight):
    img = io.imread(image)
    img = skimage.color.gray2rgb(img)
    img = transform.resize(img, (weight, hight), mode="reflect")    
    return img


for d in domains:
    path = "../dataset/visda2018/" + d + "/"
    print("processing " + path)

    if d == "train":
        f_type = "/*.png"
    elif d == "validation":
        f_type= "/*.jpg"

    for _, dirnames, _ in os.walk(path):
        dirnames.sort()
        for dirname in dirnames:
            index = dirnames.index(dirname)
            workdir = os.path.join(path, dirname)
            print(workdir)
            processed_images = io.ImageCollection(
                workdir + f_type, load_func=process_image, weight=weight, hight=hight)
            label = np.full(len(processed_images),
                            fill_value=index, dtype=np.int64)
            images = io.concatenate_images(processed_images)

            if index == 0:
                data = images
                labels = label

            else:
                data = np.vstack((data, images))
                labels = np.append(labels, label)

    print(np.shape(data))
    print(np.shape(labels))

    """
    partial = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22]
    idx = np.where(np.isin(labels, partial))
    data_p = data[idx]
    label_p = labels[idx]

    print(np.shape(data_p))
    print(np.shape(label_p))

    np.savez("../dataset/visda2018/"+d+"13.npz",
             data=data_p, label=label_p)
    print("Saved {}10.npz. It's length is {}".format(d, len(labels[idx])))
    """
    np.savez("../dataset/visda2018/"+d+".npz", data=data, label=labels)
    print("Saved {}31.npz. It's length is {}".format(d, len(labels)))
