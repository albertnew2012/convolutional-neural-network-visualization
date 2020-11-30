import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm.autonotebook import tqdm


def update_image(input_img, target_output, submodel, optimizer):
    """ gradient descent to reconstruct image
    """
    with tf.GradientTape() as tape:
        tape.watch(input_img)
        # get output from model
        outputs = submodel(input_img)
        # create gram matrix
        input_gram = style_matrix(outputs)
        input_gram = tf.reshape(input_gram, (-1))
        target_gram = style_matrix(target_output)
        target_gram = tf.reshape(target_gram, (-1))
        # get loss
        loss_value = tf.losses.mean_squared_error(input_gram, target_gram)
        grads = tape.gradient(loss_value, input_img)
        # optimize
        optimizer.apply_gradients(zip([grads], [input_img]))
        # clip data
        input_img.assign(tf.clip_by_value(input_img, 0, 1))

    return input_img, loss_value


def style_matrix(layer_out):
    """
    This function computes the style matrix, which essentially computes
    how correlated the activations of a given filter to all the other filers.
    Therefore, if there are C channels, the matrix will be of size C x C
    """
    n_channels = layer_out.get_shape().as_list()[-1]
    unwrapped_out = tf.reshape(layer_out, [-1, n_channels])
    style_matrix = tf.matmul(unwrapped_out, unwrapped_out, transpose_a=True)
    return style_matrix


def synthesize_image(input_img, target_img, submodel, steps=1000, lr=1e-2):
    # initialize optimizer
    optimizer = tf.keras.optimizers.Adam(lr)
    # get correct, output for filter
    target_output = submodel(target_img)

    loss_values = []
    # loop through gradient descent steps
    for _ in tqdm(np.arange(steps), leave=False):
        input_data, loss_value = update_image(input_img, target_output, submodel, optimizer)
        loss_values.append(loss_value.numpy())
    return np.squeeze(input_img.numpy()), loss_values


if __name__ == '__main__':
    target_img = load_img("data/painting.jpg", target_size=(244, 244, 3))
    target_img = img_to_array(target_img) / 255.
    target_img = np.expand_dims(target_img, axis=0)
    input_img = load_img("data/dog.jpeg", target_size=(244, 244, 3))
    input_img = img_to_array(input_img) / 255.
    input_img = np.expand_dims(input_img, axis=0)
    input_img = tf.Variable(input_img)

    # load VGG16 model
    model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True, classes=1000)
    conv_layers = ["block1_conv1", "block1_conv2", "block2_conv1", "block2_conv2", "block3_conv1", "block3_conv2",
                   "block3_conv3", "block4_conv1", "block4_conv2", "block4_conv3", "block5_conv1", "block5_conv2",
                   "block5_conv3", ]

    layer_imgs = []
    losses_list = []

    for layer_name in tqdm(conv_layers, leave=False):
        print(layer_name)
        submodel = tf.keras.models.Model([model.inputs[0]], [model.get_layer(layer_name).output])
        # reconstruct image
        synthesized_img, losses = synthesize_image(input_img, target_img, submodel, steps=100, lr=1e-1)
        losses_list.append(losses)
        layer_imgs.append((layer_name, np.squeeze(synthesized_img)[:, :, ::-1]))

        # plot image
        fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
        axs[0].loglog(losses)
        axs[0].set_title("loss by step")
        axs[1].imshow(np.squeeze(target_img))
        axs[1].set_title("style image")
        axs[2].imshow(np.squeeze(synthesized_img))
        axs[2].set_title(f"synthesized image from {layer_name}")
        plt.savefig(f"output/visualize_Style_transfer/{layer_name}.png")
