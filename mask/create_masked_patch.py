import json
import os
import clip
import PIL
import torch
from utils import targetpad_transform
import torch.nn.functional as F

import torch.utils.checkpoint as checkpoint
from lavis.models import load_model_and_preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
from image_masked import add_masked_patch, add_atten_masked
from torchvision import transforms


def get_masked_index_blip2(img, blip_model):
    with torch.no_grad():
        x = blip_model.visual_encoder.patch_embed(img)
        batch_size, seq_len, _ = x.size()

        cls_tokens = blip_model.visual_encoder.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if blip_model.visual_encoder.pos_embed is not None:
            x = x + blip_model.visual_encoder.pos_embed
        x = blip_model.visual_encoder.pos_drop(x)

        rel_pos_bias = blip_model.visual_encoder.rel_pos_bias() if blip_model.visual_encoder.rel_pos_bias is not None else None

        block_num = 38

        for i in range(block_num - 1):
            if blip_model.visual_encoder.use_checkpoint:
                x = checkpoint.checkpoint(blip_model.visual_encoder.blocks[i], x, rel_pos_bias)
            else:
                x = blip_model.visual_encoder.blocks[i](x, rel_pos_bias)

        atten = blip_model.visual_encoder.blocks[37].attn

        B, N, C = x.shape
        qkv_bias = None
        if atten.q_bias is not None:
            qkv_bias = torch.cat((atten.q_bias, torch.zeros_like(atten.v_bias, requires_grad=False), atten.v_bias))
        # qkv = atten.qkv(x).reshape(B, N, 3, atten.num_heads, C // atten.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=atten.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, atten.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * atten.scale
        attn = (q @ k.transpose(-2, -1))

        if atten.relative_position_bias_table is not None:
            relative_position_bias = \
                atten.relative_position_bias_table[atten.relative_position_index.view(-1)].view(
                    atten.window_size[0] * atten.window_size[1] + 1,
                    atten.window_size[0] * atten.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = atten.attn_drop(attn)

        attn = attn.sum(dim=1) / attn.shape[1]
        sub_tensor = attn[0][0][1:]
        # print("sub_tensor", attn[0][0])
        # 使用torch.topk()函数查找权重最大的192个元素的索引
        topk_values, topk_indices = torch.topk(sub_tensor, k=230)
    return topk_indices


def get_masked_index_clip(img, clip_model):
    """
        Get the output of second-to-last layer of CLIP visual encoder
    """

    with torch.no_grad():
        x = clip_model.visual.conv1(img)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                   dtype=x.dtype, device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + clip_model.visual.positional_embedding.to(x.dtype)
        x = clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        layer = clip_model.visual.transformer.layers - 1

        for i in range(layer):
            x = clip_model.visual.transformer.resblocks[i](x)
        clip_last_layer = clip_model.visual.transformer.resblocks[layer]

        attn = clip_last_layer.attn(clip_last_layer.ln_1(x), clip_last_layer.ln_1(x), clip_last_layer.ln_1(x),
                                    need_weights=True, attn_mask=None)[1]
        sub_tensor = attn[0][0][1:]
        topk_values, topk_indices = torch.topk(sub_tensor, k=218)

    return topk_indices


def get_masked_patch(path, img_id, model, transform):
    img_path = os.path.join(path, f"test/ILSVRC2012_test_{img_id}.JPEG")
    with open(img_path, 'rb') as f:
        img = PIL.Image.open(f)
        img = img.convert('RGB')

    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

    # resize = transforms.Compose([transforms.Resize((224, 224))])
    # masked_img = resize(img.copy())
    if transform:
        img_1 = transform(img).unsqueeze(0).to(device)
    index = get_masked_index_blip2(img_1, model)
    # index = get_masked_index_clip(img_1, model)
    # masked_img = add_atten_masked(masked_img, index.tolist())
    # plt.imshow(masked_img)
    # plt.axis('off')
    # plt.show()

    return index.tolist()


if __name__ == '__main__':
    with open("./img_caption_blip2.json", 'r') as f:
        img_data = json.load(f)
    print("img_data", len(img_data))

    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)

    img_caption_masked_clip = []

    blip_model, _, txt_processors = load_model_and_preprocess(name="blip2_cir_align_prompt", model_type="pretrain",
                                                              is_eval=False, device=device)
    blip_model = blip_model.eval().float()

    # clip_model, clip_preprocess = clip.load('ViT-L/14', device=device, jit=False)
    # clip_model = clip_model.eval().float()

    imageNet_path = " "

    for index in range(len(img_data)):
        img_id = img_data[index]["img_id"]
        caption = img_data[index]["caption"]
        # print("img_data[index]", img_data[index])
        index = get_masked_patch(imageNet_path, img_id, blip_model, preprocess)
        # indices = torch.randperm(256)[:230]

        s = dict()
        s["img_id"] = img_id
        s["caption"] = caption
        s["masked_index"] = index
        img_caption_masked_clip.append(s)
        print("img_id", img_id)

    print("img_caption_masked_test", len(img_caption_masked_clip))
    with open("img_caption_masked_blip_90.json", "w") as json_file:
        json.dump(img_caption_masked_clip, json_file)
