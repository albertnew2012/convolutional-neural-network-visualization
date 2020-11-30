import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.filters import gaussian
from tensorflow.keras.applications.vgg16 import decode_predictions
from tqdm.autonotebook import tqdm


def zero_one_norm(x):
    return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))


def z_score(x, scale=1.0):
    return (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) / scale)


def norm(x):
    return zero_one_norm(z_score(x))


def soft_norm(x, n_std=15):
    """ zscore and set n_std range of 0-1, then clip
    """
    x = z_score(x) / (n_std * 2)
    return tf.clip_by_value(x + 0.5, 0, 1)


def adjust_hsv(imgs, sat_exp=2.0, val_exp=0.5):
    """ normalize color for less emphasis on lower saturation
    """
    # convert to hsv

    hsv = tf.image.rgb_to_hsv(imgs)
    hue, sat, val = tf.split(hsv, 3, axis=2)

    # manipulate saturation and value
    sat = tf.math.pow(sat, sat_exp)
    val = tf.math.pow(val, val_exp)
    # rejoin hsv
    hsv_new = tf.squeeze(tf.stack([hue, sat, val], axis=2), axis=3)

    # convert to rgb
    rgb = tf.image.hsv_to_rgb(hsv_new)
    return rgb


def gen_noise(nex=1, dim=224):
    """ Generate some noise to initialize
    """
    input_img_data = tf.random.uniform((nex, dim, dim, 3))
    return tf.Variable(input_img_data, tf.float32)


def opt(submodel, optimizer, filter_index, epochs, steps=100):
    losses = []
    output_images = []
    img_data = gen_noise(len(filter_index))
    for i in tqdm(range(epochs), leave=False):
        img_data.assign([gaussian(img, multichannel=True).astype(np.float32) for img in img_data.numpy()])
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img_data)
                outputs = submodel(tf.nn.sigmoid(img_data))
                loss_value = tf.gather_nd(outputs, indices=filter_index, batch_dims=0, name=None)
                grads = tape.gradient(loss_value, img_data)
                normalized_grads = tf.math.l2_normalize(grads, axis=(1, 2, 3))
                optimizer.apply_gradients(zip([-normalized_grads], [img_data]))
                img_data.assign([z_score(img) for img in img_data.numpy()])

        losses.append(np.mean([loss_value]))
        output_images.append(img_data.numpy())

    return output_images, losses


def get_class_index(classes, class_list):
    """ grabs the index in the predication layer of the network
        based on the class name
    """
    filter_index = [np.where(class_list == i)[0][0] for i in classes]
    return filter_index


def display_features(output_images, filter_titles=None, ncols=10, zoom=5, sat_exp=2.0, val_exp=1.0):
    nrows = int(np.ceil(len(output_images[-1]) / ncols))
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * zoom, nrows * zoom))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=.95, hspace=0.1, wspace=0.01)
    for axi, ax in enumerate(axs.flatten()):
        if filter_titles is not None:
            if axi < len(filter_titles):
                ax.set_title(filter_titles[axi], fontsize=20)
        ax.axis('off')

    for i in range(len(output_images[-1])):
        ax = axs.flatten()[i]
        rgb = adjust_hsv(output_images[-1][i], sat_exp=sat_exp, val_exp=val_exp)
        pt = ax.imshow(rgb)
    plt.show()
    plt.savefig("output/visualize_classes/classes.png")


if __name__ == '__main__':
    # Create a connection between the input and the target layer
    model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True, classes=1000)

    classes = pd.DataFrame(
        np.flip(np.array(decode_predictions(np.expand_dims(np.arange(1000), 0), top=1000)[0]).reshape(-1, 3), 0),
        columns=["", "label", "index"])

    classes.columns = ["", "label", "index"]
    classes['index'] = np.int32(classes['index'].values)
    classes = classes.sort_values(by="index")
    class_list = classes.label.values

    filter_names = ['French_bulldog', 'bubble', 'Persian_cat', 'violin', 'teddy', 'mountain_bike', 'mushroom',
                    'umbrella']
    filter_index = get_class_index(filter_names, class_list)
    filter_index = [[i, j] for i, j in enumerate(filter_index)]
    layer_name = "predictions"
    # get module of input/output
    submodel = tf.keras.models.Model([model.inputs[0]], [model.get_layer(layer_name).output])
    optimizer = tf.keras.optimizers.Adam(0.1)
    output_images, losses = opt(submodel, optimizer, filter_index, epochs=5, steps=20)
    output_images = [tf.nn.sigmoid(i).numpy()[:, :, :, ::-1] for i in output_images]
    display_features(output_images, filter_names, ncols=4, zoom=4)
