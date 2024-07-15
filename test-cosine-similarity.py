import os
import torch
import clip
from PIL import Image
import warnings
from colorama import Fore, Style, init

warnings.filterwarnings('ignore')
init(autoreset=True)

def load_model(mod_type):
    if mod_type == 1:
        import shuffletxtclip as shuff_clip
    elif mod_type == 2:
        import shufflevisclip as shuff_clip
    elif mod_type == 3:
        import shuffleclip as shuff_clip
    elif mod_type == 4:
        import lobotomyclip as shuff_clip
    else:
        raise ValueError("Invalid modification type")
    return shuff_clip

def prompt_user():
    while True:
        print(f"{Style.BRIGHT}{Fore.YELLOW}\nSelect the modification to apply:\n")
        print(f"{Style.BRIGHT}{Fore.GREEN}1.{Fore.RESET} Shuffle intermediate Text Transformer layers only")
        print(f"{Style.BRIGHT}{Fore.GREEN}2.{Fore.RESET} Shuffle intermediate Vision Transformer layers only")
        print(f"{Style.BRIGHT}{Fore.GREEN}3.{Fore.RESET} Shuffle both ViT and Text intermediate layers")
        print(f"{Style.BRIGHT}{Fore.GREEN}4.{Fore.RESET} Remove every other layer between input and final in ViT + Text")
        try:
            mod_type = int(input(f"Enter a number {Style.BRIGHT}{Fore.GREEN}(1-4){Fore.RESET}: "))
            if mod_type in [1, 2, 3, 4]:
                print("\n")
                return mod_type
            else:
                print(f"{Style.BRIGHT}{Fore.RED}Invalid selection. Please try again.{Fore.RESET}")
        except ValueError:
            print(f"{Style.BRIGHT}{Fore.RED}Invalid input. Please enter a number (1-4).{Fore.RESET}")

mod_type = prompt_user()

shuff_clip = load_model(mod_type)

# Select a CLIP model to use here:
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
shuff_model, shuff_preprocess = shuff_clip.load("ViT-L/14", device=device)

# Load the text prompts -- use hardtexts.txt for confusing and tangential multi-token text labels
with open("texts.txt", "r") as f:
    texts = [line.strip() for line in f.readlines()]

text_tokens = clip.tokenize(texts).to(device)
shuff_text_tokens = shuff_clip.tokenize(texts).to(device)

# Load the images - feel free to add your own (don't forget to add a text label, too, if you do!):
image_dir = "images"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

def cosine_similarity(model, preprocess, image_files, text_tokens, device):
    similarities = []
    for image_file in image_files:
        image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)
            
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        cosine_sim = (image_features @ text_features.T).squeeze(0)
        similarities.append(cosine_sim.cpu().numpy())
    
    return similarities

original_similarities = cosine_similarity(model, preprocess, image_files, text_tokens, device)
shuffled_similarities = cosine_similarity(shuff_model, shuff_preprocess, image_files, shuff_text_tokens, device)

for image_file, orig_sim, shuff_sim in zip(image_files, original_similarities, shuffled_similarities):
    image_name = os.path.basename(image_file)
    for text, orig_val, shuff_val in zip(texts, orig_sim, shuff_sim):
        delta = abs(shuff_val - orig_val)
        if delta < 0.1:
            delta_color = f"{Style.BRIGHT}{Fore.CYAN}"
        elif delta < 0.2:
            delta_color = f"{Style.BRIGHT}{Fore.YELLOW}"
        else:
            delta_color = f"{Style.BRIGHT}{Fore.RED}"
        
        print(f"{image_name} - {Style.BRIGHT}{Fore.MAGENTA}{text}:".ljust(70) +
              f"{Style.BRIGHT}{Fore.GREEN}Original {orig_val:.3f}, ".ljust(30) +
              f"{Style.BRIGHT}{Fore.BLUE}Shuffled {shuff_val:.3f} ".ljust(30) +
              f"{delta_color}Delta {delta:.3f}")