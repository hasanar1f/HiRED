import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def show_images_with_masks(images, masks):
    n = images.shape[0]
    grid_size = 24
    patch_size = images.shape[-1] // grid_size  # 336 // 24 = 14
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        axes[i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axes[i].set_title('Full' if i == 0 else f'Sub-{i}', fontsize=30)
        axes[i].axis('off')
        mask = masks[i].reshape(grid_size, grid_size)
        for row in range(grid_size):
            for col in range(grid_size):
                if mask[row, col]:
                    rect = patches.Rectangle((col * patch_size, row * patch_size), patch_size, patch_size, linewidth=2, edgecolor='r', facecolor='none')
                    axes[i].add_patch(rect)
    
    plt.tight_layout()
    plt.show()


def plot_attn_map(attn, stride=1):
    attn = torch.mean(attn, dim=0)  # Shape: (n_tokens, n_tokens)
    attn = torch.nn.functional.avg_pool2d(attn.unsqueeze(0).unsqueeze(0), stride, stride).squeeze(0).squeeze(0)
    attn = attn.cpu().detach().numpy()  

    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(5, 5))
    log_norm = LogNorm(vmin=attn[-1].min(), vmax=attn[-1].max())
    ax = sns.heatmap(attn, cmap=cmap, norm=log_norm)
    plt.savefig('VLM_attention_map.pdf', format='pdf')
    plt.show()


def plot_heatmap(tensor,file_name):
    """
    Plot a heatmap of the input tensor and save it to a file if file_name is provided.
    """
    if tensor.dim() != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("Input tensor must be of shape [n, n]")
    
    tensor = tensor.cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(tensor, cmap='viridis', cbar=True)
    plt.title("Heatmap of Tensor")
    plt.xlabel("Index")
    plt.ylabel("Index")
    
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()



def get_random_image(h=400,w=400,show=False):
    url = f"https://picsum.photos/{h}/{w}"
    response = requests.get(url)
    if response.status_code == 200:
        with open('rand_img.jpg', 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to retrieve image. HTTP Status code: {response.status_code}")
    raw_image = Image.open('rand_img.jpg')
    if show:
        plt.imshow(raw_image)
        plt.show()
    return raw_image



def plot_attentin_map_in_nxn_grid(attention_scores):
    """
    Plot the attention map in a n x n grid
    assume num_token_patches is a perfect square
    attention_scores: torch tensor [num_token_patches]
    """
    num_tokens = attention_scores.shape[0]
    num_token_grid = int(np.sqrt(num_tokens))
    attention_scores = attention_scores.reshape((num_token_grid, num_token_grid))
    plt.figure(figsize=(5, 5))
    sns.heatmap(attention_scores, annot=False, cmap='viridis',cbar=False)
    plt.show()

def count_top_npercents_contribution(vit_attention, topk=100):
    """
    Count the contribution of topk tokens to the cls token in the attention map.
    vit_attention: tuple of torch tensor [1, num_heads, n_all_tokens, n_all_tokens]
    topk: int top k tokens to consider
    """
    num_layers = len(vit_attention)
    n_all_tokens = vit_attention[0].shape[-1]
    
    for layer in range(num_layers):
        attn = vit_attention[layer][0] # [num_heads, n_all_tokens, n_all_tokens]
        attn = attn.sum(dim=0) # [n_all_tokens, n_all_tokens]
        cls_attn = attn[0, 1:] # [n_all_tokens]
        
        cls_attn_sorted, _ = torch.sort(cls_attn, descending=True)
        # sum of topk tokens
        cls_attn_topk_sum = cls_attn_sorted[:topk].sum()
        # sum of all tokens
        cls_attn_sum = cls_attn.sum()

        # attein by topk
        print(f"Layer {layer}: {cls_attn_topk_sum/cls_attn_sum*100:.2f}%")