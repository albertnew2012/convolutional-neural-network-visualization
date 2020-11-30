import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm.autonotebook import tqdm


def gen_noise(dim=244, nex=1):
    """ Generate some noise to initialize
    """
    input_img_data = tf.random.uniform((nex, dim, dim, 3))
    return tf.Variable(tf.cast(input_img_data, tf.float32))


def update_image_step(input_data, correct_output, submodel, optimizer):
    """ gradient descent to reconstruct image
    """
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        # get output from model
        outputs = submodel(input_data)
        # get loss
        # loss_value = bce(correct_output, outputs)
        loss_value = tf.reduce_mean(tf.math.square(correct_output - outputs))
        # compute gradients
        grads = tape.gradient(loss_value, input_data)
        # optimize
        optimizer.apply_gradients(zip([grads], [input_data]))
        # clip data
        input_data.assign(tf.clip_by_value(input_data, 0, 1))

    return input_data, loss_value


def synthesize_image(img, submodel, steps=1000, lr=1e-2):
    """
    """
    # initialize optimizer
    optimizer = tf.keras.optimizers.Adam(lr)
    # generate noise
    input_data = gen_noise()
    # get correct, output for filter
    correct_output = submodel(img)
    loss_values = []
    # loop through gradient descent steps
    for _ in tqdm(np.arange(steps), leave=False):
        input_data, loss_value = update_image_step(input_data, correct_output, submodel, optimizer)
        loss_values.append(loss_value.numpy())
    return np.squeeze(input_data.numpy()), loss_values


if __name__ == '__main__':
    # Create a connection between the input and the target layer
    model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True, classes=1000)
    layers = ["block1_conv1", "block1_conv2", "block2_conv1", "block2_conv2", "block3_conv1", "block3_conv2",
              "block3_conv3", "block4_conv1", "block4_conv2", "block4_conv3", "block5_conv1", "block5_conv2",
              "block5_conv3", "fc1", "fc2", "predictions"]

    img = load_img("data/dog.jpeg", target_size=(244, 244, 3))
    img = img_to_array(img) / 255.

    layer_imgs = []
    losses_list = []
    for layer_name in tqdm(layers):
        print(layer_name)
        submodel = tf.keras.models.Model([model.inputs[0]], [model.get_layer(layer_name).output])
        # reconstruct image
        synthesized_img, losses = synthesize_image(tf.expand_dims(img[:, :, ::-1], 0), submodel, steps=500, lr=1e-2)
        losses_list.append(losses)
        layer_imgs.append((layer_name, np.squeeze(synthesized_img)[:, :, ::-1]))

        # plot image
        fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
        axs[0].loglog(losses)
        axs[0].set_title("loss by step")
        axs[1].imshow(np.squeeze(img))
        axs[1].set_title("original image")
        axs[2].imshow(np.squeeze(synthesized_img)[:, :, ::-1])
        axs[2].set_title(f"synthesized image from {layer_name}")
        plt.savefig(f"output/visualize_reconstruction/{layer_name}.png")
