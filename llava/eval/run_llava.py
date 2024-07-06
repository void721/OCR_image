import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

def visualize_attention(multihead_attention,output_path="atten_map_1.png",title="Layer 5"):
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    
    # 
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()# Shape: (n_tokens, n_tokens)
    
    
    # pooling the attention scores with stride 20
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0)
    
    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(5, 5),dpi=400)

    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    # set the x and y ticks to 20x of the original


    ax = sns.heatmap(averaged_attention,
                cmap=cmap,  # custom color map
                norm=log_norm,  # 
                # cbar_kws={'label': 'Attention score'},
                )
    
    # remove the x and y ticks
    
    # replace the x and y ticks with string

    x_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    y_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_yticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # change the x tinks font size
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    
    # make y label vertical
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)     
    
    plt.title(title)
    # tight layout
   
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()

    top_five_attentions = []
    for row in averaged_attention:
        # Use torch.topk to get the top 5 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)
        
    
    # import pdb
    # pdb.set_trace()
    
    return top_five_attentions,averaged_attention    


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    qs_noimg = qs
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs_noimg)
    conv.append_message(conv.roles[1], None)
    prompt_noimg = conv.get_prompt()
    
    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    
    input_ids_noimg = (
        tokenizer_image_token(prompt_noimg, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    

    
    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids_noimg,
    #         # images=images_tensor,
    #         # image_sizes=image_sizes,
    #         do_sample=True if args.temperature > 0 else False,
    #         temperature=args.temperature,
    #         top_p=args.top_p,
    #         num_beams=args.num_beams,
    #         max_new_tokens=args.max_new_tokens,
    #         use_cache=True,
    #     )
    # outputs_noimg = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(f"outputs_noimg:{outputs_noimg}")
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
    # import pdb
    # pdb.set_trace()
    
    outputs_attention = [output_ids['attentions']]
    
    outputs = tokenizer.batch_decode(output_ids['sequences'], skip_special_tokens=True)[0].strip()
    
    print(f"outputs_img:{outputs}")
    
    
    output_path = f"attention_result/{model_name}"
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path, exist_ok=True)
            print("Directory created successfully.")
        except Exception as e:
            print(f"Failed to create directory: {e}")
    else:
        print("Directory already exists.")
        
    # with open(output_path+"/output.json","w") as f:
    #     # json dumps
    #     json.dump({"prompt":pargs.prompt,"image":pargs.image_path,"output": model_output_ori},f,indent=4)

    # draw attention maps
    # 这样打印的是所有output输出完毕后（最后一个token generate之后的）attention map
    # 我们现在应该去考虑 每一个的attention map
    total_layers = model.config.num_hidden_layers
    for i in outputs_attention:
        for j in range(0,total_layers):
            print(f"plotting {j} layers")
            top5_attention,average_attentions = visualize_attention(i[0][j].cpu(),output_path=output_path+"/atten_map_"+str(j)+".png",title="Layer "+str(j+1))
        
    

    # stopping_criteria_id = 2 #</s> -- eos
    
    # max_output_len = 200
    # output_ids = []
    # with torch.inference_mode():
    

    #     while True:
    #         output = model(
    #             input_ids,
    #             images=images_tensor,
    #             image_sizes=image_sizes,
    #             # do_sample=True if args.temperature > 0 else False,
    #             # temperature=args.temperature,
    #             # top_p=args.top_p,
    #             # num_beams=args.num_beams,
    #             # max_new_tokens=args.max_new_tokens,
    #             # use_cache=True,
    #         )
    #         logits = output.logits
    #         next_token_logit = torch.softmax(logits[:, -1, :], dim=-1)
    #         next_token_ids = torch.argmax(next_token_logit, dim=-1)
            
    #         # print(f"debug input_ids {input_ids.shape}")
    #         # print(f"debug next_token_ids {next_token_ids.unsqueeze(0).shape}")
            
    #         input_ids = torch.cat([input_ids, next_token_ids.unsqueeze(0)], dim=-1)
            
    #         output_ids.append(next_token_ids[0].tolist())
            
    #         cur_output_len = len(output_ids)
    #         if next_token_ids == stopping_criteria_id or cur_output_len > max_output_len:
    #             break
        
        
    # output_ids = [output_ids]

    # print(f"debug output_ids {output_ids}")
    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(f"debug outputs {outputs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    # eval_model(args)
