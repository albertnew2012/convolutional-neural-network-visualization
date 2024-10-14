import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# Normalizing image would make a huge difference
def normalize(image):
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-5)


def sort_features(feature_map):
    idx = np.argsort(-1 * np.sum(feature_map, axis=(0, 1)))
    feature_map = feature_map[:, :, idx]
    return feature_map


def view_layer(feature_maps, layers_name, input_img):
    if type(feature_maps) is not list:
        feature_maps = [feature_maps]
        layers_name = [layers_name]
    for i in range(len(layers_name)):
        feature_map = feature_maps[i][0].numpy()
        layer_name = layers_name[i]
        feature_map = sort_features(feature_map)
        # create an empty canvas
        feature_size = 8 * np.array(feature_map[:, :, 0].shape)
        features = np.zeros(feature_size)
        activation_size = 8 * np.array(input_img.shape[1:3])
        activation = np.zeros(activation_size)
        img_gray = rgb2gray(input_img[0])
        h, w = feature_map.shape[0], feature_map.shape[1]
        H, W = img_gray.shape[0], img_gray.shape[1]
        for i in range(64):
            x = i % 8
            y = i // 8
            features[y * h:(y + 1) * h, x * w:(x + 1) * w] = feature_map[:, :, i]
            feature_map_resized = resize(normalize(feature_map[:, :, i]), input_img.shape[1:3])
            activation[y * H:(y + 1) * H, x * W:(x + 1) * W] = feature_map_resized * img_gray
        plt.figure()
        plt.title(f"feature maps: {layer_name}")
        plt.axis("off")
        plt.imshow(features)
        plt.savefig(f"output/visualize_activation/feature_{layer_name}.png")
        plt.show()
        plt.close()  # Close figure after displaying

        plt.figure()
        plt.title(f"activation {layer_name}")
        plt.imshow(activation)
        plt.axis("off")
        plt.show()
        plt.savefig(f"output/visualize_activation/activation_{layer_name}.png")
        plt.close()  # Close figure after displaying

def load_model(model, layers_name):
    if type(layers_name) is str:
        layers_name = [layers_name]
    submodel = Model([model.inputs[0]], [model.get_layer(layer_name).output for layer_name in layers_name])
    return submodel


if __name__ == '__main__':
    # load model
    model = VGG16(weights='imagenet', include_top=True)
    layers_name = [layer.name for layer in model.layers if "conv" in layer.name]
    submodel = load_model(model, layers_name)
    IMAGE_PATH = 'data/dog.jpeg'  # or 'data/cat.jpg'
    img = load_img(IMAGE_PATH, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    feature_maps = submodel(img)

    # show the feature map and activation detected by each convNet layer
    view_layer(feature_maps, layers_name, img)

    # show the object being detected in by the last convNet layer in sumation
    feature_sum = np.sum(feature_maps[-1][0], axis=-1)
    feature_sum_resize = resize(normalize(feature_sum), (224, 224))
    output = feature_sum_resize[..., np.newaxis] * img[0]
    plt.figure()
    plt.imshow(feature_sum)
    plt.savefig("output/visualize_activation/feature_sum.png")
    plt.figure()
    plt.imshow(output.astype(np.uint8))
    plt.savefig("output/visualize_activation/detection_sum.png")
    # plt.close("all")
    plt.show()

    # show the object being detected in by the last convNet layer
    layers_name = [layer.name for layer in model.layers if "fc" in layer.name]
    submodel = load_model(model, layers_name)
    output_fc1,output_fc2 = submodel(img)
    output_fc2_img = normalize(output_fc2.numpy()).reshape(64,64)
    plt.figure()
    plt.imshow(output_fc2_img)
    plt.savefig("output/visualize_activation/fc2.png")
    plt.show()
