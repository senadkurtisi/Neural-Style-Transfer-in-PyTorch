from utils.globals import *
from utils.utils import load_image, show_image

from style_transfer import NST

if __name__ == "__main__":
    # Load content image
    content_img = load_image(config.content_loc, config, device)
    # Load style image
    style_img = load_image(config.style_loc, config, device)

    # Show loaded images
    show_image(content_img, 'Content image', should_save=False)
    show_image(style_img, 'Style image', should_save=False)

    # Perform the Neural Style Transfer
    generated_image = NST(content_img, style_img, config, device)
    # Show the synthetized image & save it to the ouput folder
    show_image(generated_image, title='Synthetized image', should_save=True)
