### CLIP ViT Layer Deck Shuffle ♻️🤖⁉️
----
### Update 24/SEPT/2024:
New `shuffle-comp-gradient-ascent.py`, usage:
- `python shuffle-comp-gradient-ascent.py --image_path images/cat.jpg`
- Edit code (see #comments) to configure:
- layer_range [1, 4, 5, 6] to shuffle, visual or text transformer
- Shuffle whole layer OR just attn OR just MLP or None
- Or set Layer Normalization to nn.identity (no-op)
- Optional: Manipulate individual neuron activation values.
- ⚠️ Default config leads CLIP to (nonsensically) cuss about *any* input image.
- ⚠️ Use responsibly & enjoy exploring CLIP's bias! 🤗

![clipshuffle-git](https://github.com/user-attachments/assets/1a38bb51-93a9-4959-8697-ba6aa7c4e8ca)
----
Previous update / initial commit:
----
![example-github](https://github.com/user-attachments/assets/b6955be7-670b-4859-a74d-22fbb9d1e155)


- Inspired by the paper [Transformer Layers as Painters](https://arxiv.org/abs/2407.09298v1), which focuses on LLM transformers.
- This code applies similar experimental modifications to CLIP, namely randomly shuffling the order of intermediate layers of the text transformer, the vision transformer, or both.
- Additionally, there is an option to strip every other layer between the input and output layers, effectively reducing CLIP's depth by half, while maintaining the order of the remaining layers.

- All types of modifications are detrimental to CLIP, as the model seems to require highly hierarchical feature extraction. However, the text transformer appears slightly more robust to feature shuffling compared to the vision transformer.
- A surprising result is that stripping half of CLIP's entire text and vision transformer layers is *LESS* detrimental to performance than shuffling the order of a mere 5 out of the 24 layers in ViT - although this depends on the specific text-image pair.
--------
- To run the code, execute `python test-cosine-similarity.py` and follow the interactive instructions.
- Use `model.py` from the "originalclip" folder to 'do a diff' against `model.py` in the modified folders to make your own changes.
-------
Requires [OpenAI/CLIP](https://github.com/openai/CLIP) and colorama (`pip install colorama`).

For visualizing CLIP ViT features as in [image above], see my repo [zer0int/CLIP-ViT-visualization](https://github.com/zer0int/CLIP-ViT-visualization)
