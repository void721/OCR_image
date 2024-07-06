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

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


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
    
    # llava原始代码
    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         images=images_tensor,
    #         image_sizes=image_sizes,
    #         do_sample=True if args.temperature > 0 else False,
    #         temperature=args.temperature,
    #         top_p=args.top_p,
    #         num_beams=args.num_beams,
    #         max_new_tokens=args.max_new_tokens,
    #         use_cache=True,
    #     )
    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(f"outputs1:{outputs}")


    stopping_criteria_id = 2 #</s> -- eos
    
    max_output_len = 200
    output_ids = []
    with torch.inference_mode():
    

        while True:
            
            output = model(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                output_attentions=True,
                # output_scores=True,
                # do_sample=True if args.temperature > 0 else False,
                # temperature=args.temperature,
                # top_p=args.top_p,
                # num_beams=args.num_beams,
                # max_new_tokens=args.max_new_tokens,
                # use_cache=True,
            )
            
            
            # output.keys()
            # odict_keys(['logits', 'past_key_values','attentions'])
            
            # len(output['past_key_values']) 32
            # output['logits'].shape torch.Size([1, 2189, 32000]   
            
            # 2189是总共的attention 长度
            
            
            
            logits = output.logits
            next_token_logit = torch.softmax(logits[:, -1, :], dim=-1)
            next_token_ids = torch.argmax(next_token_logit, dim=-1)
            input_ids = torch.cat([input_ids, next_token_ids.unsqueeze(0)], dim=-1)
            
            output_ids.append(next_token_ids[0].tolist())
            
            # attentions= output['attentions'].cpu()
            
            print(f"logits:{logits.shape}")
            print(f"next_token_logit:{next_token_logit.shape}")
            print(f"input_ids:{input_ids.shape}")
            
            import pdb
            pdb.set_trace()
            # 没有开attention == True的时候
            # logits:torch.Size([1, 2238, 32000])
            # next_token_logit:torch.Size([1, 32000])
            # next_token_ids:tensor([278], device='cuda:0')

            
            
            # print(f"debug input_ids {input_ids.shape}")
            # print(f"debug next_token_ids {next_token_ids.unsqueeze(0).shape}")
            
            
            
            # attn_map setting
            # 需要打印infer时候每一个token的attention map
            
            
            # output_attention = output['attentions']
            # print(f"print output_attention == size:{len(output_attention[0][0][0])},{len(output_attention[0][0][0][0])}")
            # output_past_key_values = output['past_key_values'] # (2,1,32,2189,128)
            # print(f"print output_past_key_values == siz[e:{len(output_past_key_values[0][0][0][0])},{len(output_past_key_values[0][0][0][0][0])}")
            # output_logits = output['logits'] # shape: torch.Size([1, 2189, 32000])
            # print(f"print output_logits == size:{output_logits.shape}")
            

            
            
            
            # print(f"output:past_key_values: {np.array(output_past_key_values.cpu()).shape}")
            # print(f"output:logits: {np.array(output_logits.cpu()).shape}")
            

            
            cur_output_len = len(output_ids)
            
            
            if next_token_ids == stopping_criteria_id or cur_output_len > max_output_len:
                break
        
        
    output_ids = [output_ids]

    # print(f"debug output_ids {output_ids}")
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"debug outputs {outputs}")

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

    eval_model(args)
