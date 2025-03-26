import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL.ImageOps import autocontrast
from torchvision.transforms.functional import adjust_contrast


def plot_images(original, modified, titles):
    """
    A grid where the top row is the original images and the bottom row is the modified images.
    Args:
        original (list): A list of original images
        modified (list): A list of modified images
        titles (list): A list of titles for the images
    """
    _, axes = plt.subplots(2, len(modified), figsize=(20, 10))
    for ax in axes[0]:
        ax.imshow(original, cmap="gray")
        ax.set_title("Original")
        ax.axis("off")
        ax.grid(True)

    for _, (ax, img, title) in enumerate(zip(axes[1], modified, titles)):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("results/contrast_results.png")
    plt.show()

    if not os.path.exists("results"):
        os.makedirs("results")


# --------------------------------------------------------------------------------
# METHOD 1: Using a simple filter model to learn the filter
# --------------------------------------------------------------------------------
class SimpleFilterModel(torch.nn.Module):
    def __init__(self, input_height, input_width):
        """
        A simple filter model that learns a filter to apply to the input image
        by learning the weights of the filter and convolving it with the input image.
        We can never guarantee what the filter will learn, which makes it a black box.

        For example, we may want the filter to learn to apply autocontrast to the image,
        but we can't guarantee that it will learn that. It may learn something completely
        different based on which parts of the filter are affected the most by the loss.

        Args:
            input_size (int): The size of the input image (assuming square image)
        """
        super(SimpleFilterModel, self).__init__()
        self.filter = torch.nn.Parameter(
            torch.randn(1, 1, input_height, input_width)
        )  # N filter x Input Channels x Height x Width

    def forward(self, x):
        return torch.conv2d(
            x, self.filter
        )  # Just convolve the filter with the input image


# --------------------------------------------------------------------------------
# METHOD 2: Using a more complex model to learn the filter
# --------------------------------------------------------------------------------
class AutoContraster:
    """
    A class that applies autocontrast to an image using different libraries. The two
    libraries used are PIL and Torch. Try both and see which one works better!


    """

    def apply_pillow_autocontrast(self, img):
        # Refer: https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.autocontrast
        return autocontrast(img)

    def apply_torch_adjustcontrast(self, img):
        # Refer: https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_contrast.html
        return adjust_contrast(img, 3)  # 3 is the contrast factor


# --------------------------------------------------------------------------------
# METHOD 3: Manual Implementation of Autocontrast with Custom Logic
# --------------------------------------------------------------------------------
class ManualContraster:
    """
    A class that applies autocontrast to an image using custom logic. How did I come up with
    this logic? I don't know, but combining PILs histograms with some scaling seemed like a reasonable idea.
    """

    def apply_autocontrast(self, img):
        hist = img.histogram()

        cdf = [sum(hist[: i + 1]) for i in range(256)]

        # The first 45% of pixels will be clamped to 0
        # Why 45%? The cells are either the brightest or darkest pixels in the image, depending on your microscope.
        # This is an assumption, but maybe a good and helpful one.
        # If the background is light grey, you’d take the last 45% of pixels instead.
        # Here, we assume cells are the brightest.
        # Since cells always have at least one bright spot, they won't fall below the median brightness,
        # so removing the lowest 45% enhances contrast without erasing them.
        # So the logic: make a histogram, calculate the cdf and find the cutoff at 45% of the cdf
        min_five_percent = 0.45 * cdf[-1]
        cutoff = 0.45 * cdf[-1]
        for i in range(256):
            if cdf[i] >= min_five_percent:
                cutoff = i
                break

        # update image
        img = img.point(lambda p: p if p > cutoff else 0)

        # Transform the image to a tensor for further processing
        transform = transforms.Compose([transforms.PILToTensor()])
        img_tensor = transform(img).float()

        # Compute the minimum and maximum pixel values
        min_val = torch.min(img_tensor)
        max_val = torch.max(img_tensor)

        # Compute the range of pixel values
        range_val = max_val - min_val

        # Compute the scale factor
        scale = 255.0 / range_val

        # Apply the scale factor to the image
        # If the image is already vry contrasted, the scale factor will be very low, so it will not change much
        # If the image is not contrasted, the scale factor will be high, so it will change a lot
        img_tensor = (img_tensor - min_val) * scale

        # Clip the pixel values to the range [0, 255]
        img_tensor = torch.clamp(img_tensor, 0, 255)

        # Convert the image back to a PIL image
        img_tensor = img_tensor.numpy().astype("uint8")

        return img_tensor.squeeze(0)


def main():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sample_img = os.path.join(cur_dir, "data/sample.jpg")

    # Load the image as black and white
    img = Image.open(sample_img).convert("L")

    # Convert the image to a tensor
    transform = transforms.Compose([transforms.PILToTensor()])
    img_tensor = transform(img).float()

    # Method 1: Apply autocontrast using a simple filter model
    filter_model = SimpleFilterModel(
        input_height=img_tensor.shape[1], input_width=img_tensor.shape[2]
    )
    img_filter = filter_model(img_tensor)

    # Method 2a: Apply autocontrast using the PIL library
    auto_contraster = AutoContraster()
    img_pillow = auto_contraster.apply_pillow_autocontrast(img)

    # Method 2b: Apply autocontrast using the Torch library
    img_torch = auto_contraster.apply_torch_adjustcontrast(img_tensor)

    # Method 3: Apply autocontrast using a manual implementation
    manual_contraster = ManualContraster()
    img_manual = manual_contraster.apply_autocontrast(img)

    plot_images(
        img,
        [
            img_filter.squeeze(0).detach().numpy(),
            img_pillow,
            img_torch.squeeze(0).numpy(),
            img_manual,
        ],
        [
            "Filter Model (Untrained)",
            "Pillow Autocontrast (Default Settings)",
            "Torch Autocontrast (Factor 3)",
            "Manual Autocontrast (Magic)",
        ],
    )


if __name__ == "__main__":
    main()
