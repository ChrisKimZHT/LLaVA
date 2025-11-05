import re

import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.eval.run_llava import load_images
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

disable_torch_init()

model_path = "liuhaotian/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
default_query = "Describe this picture in detail."
default_image = "https://llava-vl.github.io/static/images/view.jpg"

while True:
    query = input("Prompt: ")
    if query.lower() == "q":
        break
    if len(query) == 0:
        query = default_query

    image_file = input("Image: ").removeprefix("\"").removesuffix("\"")
    if image_file.lower() == "q":
        break
    if len(image_file) == 0:
        image_file = default_image

    if IMAGE_PLACEHOLDER in query:
        query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    else:
        query = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = load_images([image_file])
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

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.6,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)
