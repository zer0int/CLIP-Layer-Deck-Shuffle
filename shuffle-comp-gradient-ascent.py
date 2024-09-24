'''
Original CLIP gradient ascent by Twitter / X @advadnoun
'''
import os
import torch
import clip
import imageio
import torchvision
import PIL.Image
checkin_step = 10
import sys
import kornia
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Resize
import random
import numpy as np
import argparse
import glob
from multiprocessing import cpu_count
from tqdm import tqdm
import warnings
import pickle
import warnings
from colorama import Fore, Style
import pdb
warnings.filterwarnings('ignore')

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

clipmodel = 'ViT-L/14' 
training_iterations = 300
batch_size = 12
many_tokens = 4
input_dims = 224

lestiforgot = 'TOK'
os.makedirs(lestiforgot, exist_ok=True)

parser = argparse.ArgumentParser(description="CLIP Gradient Ascent")
parser.add_argument("--image_path", type=str, required=True, help="Path to the input image, e.g.: --image_path images/cat.jpg")
args = parser.parse_args()

perceptor, preprocess = clip.load(clipmodel, jit=False)


shuffle_setting = "Ident"  # Options: "None", "MLP", "Layer", "Attn", "Ident"

texttransformer = False # False: Use visual transformer for shuffle instead

layer_range = [9, 10, 11, 12, 13] # ViT-L/14: 24 [0-23] in visual.transformer, 12 [0-11] in (text) transformer
#layer_range = [3, 4, 5, 6, 7] # => center layers of text transformer

# PS: Scroll all the way down -> individual neuron activation value manipulation opportunity! ;-)

def shuffle_components(perceptor, layer_range, shuffle_setting):
    for layer_idx in layer_range:
        if texttransformer:
            layer1 = perceptor.transformer.resblocks[layer_idx]
            layer2 = perceptor.transformer.resblocks[layer_idx + 1]
        if not texttransformer:
            layer1 = perceptor.visual.transformer.resblocks[layer_idx]
            layer2 = perceptor.visual.transformer.resblocks[layer_idx + 1]

        if shuffle_setting == "MLP":
            # Shuffle only the MLP components
            layer1.mlp.c_fc.weight, layer2.mlp.c_proj.weight = layer2.mlp.c_fc.weight, layer1.mlp.c_proj.weight
            layer1.mlp.c_fc.bias, layer2.mlp.c_proj.bias = layer2.mlp.c_fc.bias, layer1.mlp.c_proj.bias

        elif shuffle_setting == "Attn":
            # Shuffle only the attention components
            layer1.attn.out_proj.weight, layer2.attn.out_proj.weight = layer2.attn.out_proj.weight, layer1.attn.out_proj.weight
            layer1.attn.out_proj.bias, layer2.attn.out_proj.bias = layer2.attn.out_proj.bias, layer1.attn.out_proj.bias

        elif shuffle_setting == "Layer":
            # Shuffle the whole layer (everything in the layer)
            layer1, layer2 = layer2, layer1

        elif shuffle_setting == "Ident":
            # Set the LayerNorm to be an identity operation (no-op)
            layer1.ln_1 = torch.nn.Identity()
            layer2.ln_1 = torch.nn.Identity()
            

        elif shuffle_setting == "None":
            # Do nothing
            pass

shuffle_components(perceptor, layer_range, shuffle_setting)

perceptor = perceptor.cuda()
perceptor = perceptor.eval().float()

def displ(img, pre_scaled=True):
    img = np.array(img)[:,:,:]
    img = np.transpose(img, (1, 2, 0))
    if not pre_scaled:
        img = scale(img, 48*4, 32*4)
    imageio.imwrite(str(3) + '.png', np.array(img))
    return display.Image(str(3)+'.png')

def clip_encode_text(gobble, text):
    x = torch.matmul(text, gobble.token_embedding.weight)  # [batch_size, n_ctx, d_model]
    x = x + gobble.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = gobble.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = gobble.ln_final(x)
    x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ gobble.text_projection
    return x

prompt = clip.tokenize('''''').numpy().tolist()[0]
prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]

img_path = args.image_path
img_name = os.path.splitext(os.path.basename(img_path))[0]

im = torch.tensor(imageio.imread(img_path).copy()).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255
im = F.interpolate(im, (input_dims, input_dims))

torch.cuda.empty_cache()

class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000
        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1
        ptt = prompt
        self.prompt = torch.zeros(batch_size, len(ptt), 49408).cuda()
        for jk, pt in enumerate(ptt):
            self.prompt[:, jk, pt] = 1
        self.pad = torch.zeros(batch_size, 77 - (many_tokens + len(prompt) + 1), 49408).cuda()
        self.pad[:, :, 49407] = 1

    def forward(self):
        self.soft = F.gumbel_softmax(self.normu, tau=self.much_hard, dim=-1, hard=True)
        fin = torch.cat([self.start, self.prompt, self.soft, self.pad], 1)
        return fin

lats = Pars().cuda()
mapper = [lats.normu]
optimizer = torch.optim.Adam([{'params': mapper, 'lr': 5}])

nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

augs = torch.nn.Sequential(
    kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
).cuda()

tok = clip.simple_tokenizer.SimpleTokenizer()

bests = {1000:'None', 1001:'None', 1002:'None', 1003:'None', 1004:'None'}

def augment(into):
    into = augs(into)
    return into

# Custom hook to scale the feature activation
class FeatureScalerHook:
    def __init__(self, model, layer_idx, feature_idx, scale_factor, transformer_type='visual'):
        self.model = model
        self.layer_idx = layer_idx
        self.feature_idx = feature_idx
        self.scale_factor = scale_factor
        self.transformer_type = transformer_type
        self.handle = None
        self.register_hook()

    def register_hook(self):
        def hook(module, input, output):
            output[:, :, self.feature_idx] *= self.scale_factor
            return output

        if self.transformer_type == 'visual':
            layer = self.model.visual.transformer.resblocks[self.layer_idx].mlp.c_fc
        else:
            layer = self.model.transformer.resblocks[self.layer_idx].mlp.c_fc
        self.handle = layer.register_forward_hook(hook)

    def remove(self):
        if self.handle:
            self.handle.remove()

def ascend_txt():
    global im
    iii = nom(augment(im[:,:3,:,:].expand(64, -1, -1, -1)))
    iii = perceptor.encode_image(iii).detach()
    lll = lats()
    tx = clip_encode_text(perceptor, lll)
    return -100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, batch_size).T.mean(1), lll

def train():
    with autocast():
        loss1, lll = ascend_txt()
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, lll

def checkin(loss, lll):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('', '').replace('', '') for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}")
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('?', '')
        j = j.replace('!', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace('*', '')
        j = j.replace(',', '')

        tokens = j.split()
        unique_tokens.update(tokens)

    with open(f"TOK/tokens_{img_name}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

def loop():
    scaling_factors = [10]  # in order to conduct multiple runs with different factors for testing
    # I am sorry for the confusion re: these comments. They're examples for what changes in CLIP's output for a given neuron. :-)
    # '+' means increase activation value, typically scale_factor 100 or 1000; '-' means negative scale_factor.
    top_activations_layer_20 = [2089]#[2465, 2089] 2465 + feels reckon buds bly REKT - deplordiscriminsanitation distinguished
    # 2089: distortion glitch scrambled img glitch omfg -1000: unable to classify
    top_activations_layer_21 = [1890]# + non-sensical REKT no classification - good ?
    top_activations_layer_22 = [3289]#
    top_activations_layer_23 = [733]# 733: + ooooooo shawsabotage mlb - glitch mri rgb
    top_activations_text_layer_10 = [4, 2695, 777, 1296, 2192, 2072, 1178, 2332, 1314, 2212, 813, 321, 1091, 1352, 1097, 716, 1127, 1653, 1526, 1146, 3070, 1279]
    top_activations_text_layer_11 = [884, 2508, 1625]
    top_activations_text_layer_11_less = [2575, 441, 1332]
    #2209: + no influence, - unable to classify
    #1332: + rgb, glitch, wacky, psychedelic laser, abstr, pixels - still psyched, less able to classify
    #2508: + glitch diversity lgbtq, warp, magic, 🔮 - uhh trippy seizures, lsd
    #46:   + digi gey, abstr, rgb, digi, pixels, machinelearning jekyll scan - awful result
    #2575: + textured, interference, colorized - geist acid unfit algorithms
    #1625: + psychedelic trippy robot abstr -- but NOT pixel art for this one
    #467:  + abstrbruh cgi rgb rendered - mostly unable to classify, rekt
    #884:  + pixelart deeplearbitpsychedelic fantasy bitpixels scrambled 

    
    for factor in scaling_factors:
        print(f"\nTesting with scaling factor: {factor}\n")

        # Scale activations in visual transformer layer 20
        hooks_layer_20 = []
        for feature_idx in top_activations_layer_20:
            hook = FeatureScalerHook(perceptor, layer_idx=20, feature_idx=feature_idx, scale_factor=1, transformer_type='visual')
            hooks_layer_20.append(hook)
            
        # Scale activations in visual transformer layer 21
        hooks_layer_21 = []
        for feature_idx in top_activations_layer_21:
            hook = FeatureScalerHook(perceptor, layer_idx=21, feature_idx=feature_idx, scale_factor=1, transformer_type='visual')
            hooks_layer_21.append(hook)
            
        # Scale activations in visual transformer layer 22
        hooks_layer_22 = []
        for feature_idx in top_activations_layer_22:
            hook = FeatureScalerHook(perceptor, layer_idx=22, feature_idx=feature_idx, scale_factor=1, transformer_type='visual')
            hooks_layer_22.append(hook)
            
        # Scale activations in visual transformer layer 22
        hooks_layer_23 = []
        for feature_idx in top_activations_layer_23:
            hook = FeatureScalerHook(perceptor, layer_idx=23, feature_idx=feature_idx, scale_factor=1, transformer_type='visual')
            hooks_layer_23.append(hook)
        
        # Scale activations in text transformer layer 9
        hooks_text_layer_10 = []
        for feature_idx in top_activations_text_layer_10:
            hook = FeatureScalerHook(perceptor, layer_idx=9, feature_idx=feature_idx, scale_factor=1, transformer_type='text')
            hooks_text_layer_10.append(hook)

        # Scale activations in text transformer layer 10
        hooks_text_layer_11 = []
        for feature_idx in top_activations_text_layer_11:
            hook = FeatureScalerHook(perceptor, layer_idx=10, feature_idx=feature_idx, scale_factor=1, transformer_type='text')
            hooks_text_layer_11.append(hook)
        
        # Scale activations in text transformer layer 11
        hooks_text_layer_11_less = []
        for feature_idx in top_activations_text_layer_11_less:
            hook = FeatureScalerHook(perceptor, layer_idx=11, feature_idx=feature_idx, scale_factor=1, transformer_type='text')
            hooks_text_layer_11_less.append(hook)
        
        for i in range(training_iterations):
            loss, lll = train()
            if i % checkin_step == 0:
                checkin(loss, lll)
                #print(Fore.YELLOW + f"Iteration {i}: Average Loss: {loss.mean().item()}")

loop()


