from utils.globals import *
from utils.utils import *

from style_transfer import NST

if __name__ == "__main__":
    # Load content image
    content_img = load_image(config.content_loc, config, device)
    # Load style image
    style_img = load_image(config.style_loc, config, device)

    # Show loaded images
    show_image(content_img, 'Content image')
    show_image(style_img, 'Style image')

    # Perform the Neural Style Transfer
    generated_image = NST(content_img, style_img, config, device)
    # Show the synthetized image
    show_image(generated_image, 'Synthetized image')
