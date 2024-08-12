# PTG-FSCIR
We have open-sourced the data and code we used, which includes: 
1. The code for calculating attention masks, the module for adding attention, and the patches of attention masks we obtained from ImageNet.
2. The image captions used in the second stage on the FashionIQ, CIRR, and B2W training sets, as well as the results after our filtering and scoring.
## Our backbones
In our method, the two backbones used have already been open-sourced in previous research.\\
### SPRC

SPRC is available for download and contributions on GitHub. For more information, check out the repository:
[SPRC Repository](https://github.com/chunmeifeng/SPRC)

### CLIP4CIR

For those interested in the CLIP4CIR project, you can access its resources and contribute via its GitHub page:
[CLIP4CIR Repository](https://github.com/ABaldrati/CLIP4Cir)

## How to use


### Masked Image Data

Our mask blocks are located in the `/mask/` directory. You can use them with the library provided in the `image_masked.py` file. Below is an example function from our script that demonstrates how to load and apply masks to an image:

You can also use 'creat_masked_patches.py' to creat masked dataset on other image datasets.

Here is an example how to use masked image in your dataset code.

```python
def get_img(self, img_id, mask_list):
    img_path = os.path.join(self.path, f"test/ILSVRC2012_test_{img_id}.JPEG")
    with open(img_path, 'rb') as f:
        img = PIL.Image.open(f)
        img = img.convert('RGB')
    masked_img = self.resize(img.copy())
    masked_img = add_atten_masked(masked_img, mask_list)
```
You can apply the aforementioned process as the first stage we mentioned for the backbone.

### Score data
You can find our computed scores in the `/score/{backbone}/` directory. Each key corresponds to an index, and the sequence of these indices aligns with the public dataset captions. You are free to segment according to the scores and select randomly as needed.

you can apply the selected samples as the second stage for the backbone.
