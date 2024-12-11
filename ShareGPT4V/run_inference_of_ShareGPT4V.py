import torch
from PIL import Image
from io import BytesIO
import requests

from share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from share4v.conversation import SeparatorStyle, conv_templates
from share4v.mm_utils import (KeywordsStoppingCriteria,
                              get_model_name_from_path, tokenizer_image_token)
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init


def load_image(image_file):
    if image_file.startswith('http'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def run_inference(model_path, model_base, image_file, query, conv_mode=None):
    # Disable torch initialization for faster inference
    disable_torch_init()

    # Load model, tokenizer, and image processor
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    # Prepare prompt with image tokens if necessary
    if model.config.mm_use_im_start_end:
        prompt_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
    else:
        prompt_text = DEFAULT_IMAGE_TOKEN + '\n' + query


    conv_mode = "share4v_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    # Load and preprocess image
    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    # Tokenize the prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # Stopping criteria for the generated output
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    # Decode and process the output
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


# Define inputs
image_path = "image.jpg"
prompt = "describe the image in details"

# Run inference and print the result
model_path = "Lin-Chen/ShareGPT4V-7B"
model_base = None
output = run_inference(model_path, model_base, image_path, prompt)
print("Generated Description:", output)