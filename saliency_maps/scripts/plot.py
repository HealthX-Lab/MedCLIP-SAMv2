import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.text import Text
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

class TextWithBGColor(Text):
    def __init__(self, x, y, text, bgcolor, *args, **kwargs):
        super().__init__(x, y, text, *args, **kwargs)
        self.bgcolor = bgcolor

    def draw(self, renderer, *args, **kwargs):
        bbox = dict(facecolor=self.bgcolor, edgecolor=self.bgcolor, boxstyle='round,pad=0.01', alpha=self.bgcolor[3])
        self.set_bbox(bbox)
        super().draw(renderer, *args, **kwargs)


def plot_text_with_colors(ax, tokens, rgba_colors, max_width, max_height, fontsize=12):
    x, y = 0, max_height
    space_width = fontsize * 0.2  # Width of a single space
    for i, (token, color) in enumerate(zip(tokens, rgba_colors)):
        text = TextWithBGColor(x, y*0.7, token, color, fontsize=fontsize)
        ax.add_artist(text)
        
        token_width = fontsize * 0.4 * len(token)

        if i + 1 < len(tokens):
            next_token_width = fontsize * 0.4 * len(tokens[i + 1]) + space_width
            if x + token_width + next_token_width >= max_width:
                x = 0
                y -= 1  # Decrease the space between rows
            else:
                x += token_width + space_width  # Add the width of a single space after each token
        else:
            x += token_width

def generate_shades_with_alpha(scores, min_alpha=0, max_alpha=0.8):
    normalized_scores = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]
    red_colors = []
    for score in normalized_scores:
        alpha = min_alpha + score * (max_alpha - min_alpha)
        red_color = (0.2, 0.4, 1, alpha)
        red_colors.append(red_color)
    return red_colors

    
def visualize_vandt_heatmap(tmap, vmap, text_words, image, title=None, vtitle=None, ttitle=None, max_width=100, max_height=4):
    text_words = [x.split('<')[0] for x in text_words[1:-1]] # remove start and end tokens
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(6,3)
    #axs[0].imshow(np.float32(image)*np.expand_dims(vmap, axis=2))
    axs[0].imshow(show_cam_on_image(np.float32(image), vmap, use_rgb=True))
    # show bounding box if available
    axs[0].axis('off')
    rgba_colors = generate_shades_with_alpha(tmap[1:-1])
    plot_text_with_colors(axs[1], text_words, rgba_colors, max_width=max_width, max_height=max_height, fontsize=14)
    axs[1].set_xlim(0, max_width)
    axs[1].set_ylim(-max_height, max_height)
    axs[1].axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.tight_layout()
    if title:
        plt.savefig(title,bbox_inches='tight')