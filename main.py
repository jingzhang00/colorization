from PIL import Image
import gradio as gr
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import gc
import cv2
import numpy as np

controlnet = ControlNetModel.from_pretrained("ioclab/control_v1p_sd15_brightness", torch_dtype=torch.float16, use_safetensors=True)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    # "/models--runwayml--stable-diffusion-v1-5/snapshots/ded79e214aa69e42c24d3f5ac14b76d568679cc2",
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()

def infer(
        prompt,
        negative_prompt,
        conditioning_image,
        num_inference_steps=30,
        size=768,
        guidance_scale=7.0,
        seed=1234,
):

    conditioning_image_raw = Image.fromarray(conditioning_image)
    conditioning_image = conditioning_image_raw.convert('L')

    g_cpu = torch.Generator()

    if seed == -1:
        generator = g_cpu.manual_seed(g_cpu.seed())
    else:
        generator = g_cpu.manual_seed(seed)

    output_image = pipe(
        prompt,
        conditioning_image,
        height=size,
        width=size,
        num_inference_steps=num_inference_steps,
        generator=generator,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=1.0,
    ).images[0]

    # restore origin size
    conditioning_image = np.array(conditioning_image)
    output_image = np.array(output_image)

    conditioning_image =  cv2.merge([conditioning_image, conditioning_image, conditioning_image])
    h, w, _ = conditioning_image.shape
    y = cv2.cvtColor(conditioning_image, cv2.COLOR_RGB2YUV)[:, :, 0]

    output_image = cv2.resize(output_image, (w, h))
    output_image_yuv = cv2.cvtColor(output_image, cv2.COLOR_RGB2YUV)

    output_image_yuv[:, :, 0] = y
    output_image_orisz = cv2.cvtColor(output_image_yuv, cv2.COLOR_YUV2RGB)

    del conditioning_image, conditioning_image_raw
    gc.collect()

    return Image.fromarray(output_image_orisz)

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Diffusion-based Colorization
    This is a colorization demo based on diffusion.
    """)

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
            )
            conditioning_image = gr.Image(
                label="Conditioning Image",
            )
            with gr.Accordion('Advanced options', open=False):
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        10, 40, 20,
                        step=1,
                        label="Steps",
                    )
                    size = gr.Slider(
                        256, 768, 512,
                        step=128,
                        label="Size",
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label='Guidance Scale',
                        minimum=0.1,
                        maximum=30.0,
                        value=7.0,
                        step=0.1
                    )
                    seed = gr.Slider(
                        label='Seed',
                        value=-1,
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        # randomize=True
                    )
            submit_btn = gr.Button(
                value="Submit",
                variant="primary"
            )
        with gr.Column(min_width=300):
            output = gr.Image(
                label="Result",
            )

    submit_btn.click(
        fn=infer,
        inputs=[
            prompt, negative_prompt, conditioning_image, num_inference_steps, size, guidance_scale, seed
        ],
        outputs=output
    )
    gr.Examples(
        examples=[
            ["a village in the mountains", "monochrome", "/home/omen/Desktop/colorization/conditioning_images/conditioning_image_1.jpg"],
            ["three people walking in an alleyway with hats and pants", "monochrome", "/home/omen/Desktop/colorization/conditioning_images/conditioning_image_2.jpg"],
            ["an anime character, natural skin", "monochrome, blue skin, grayscale", "/home/omen/Desktop/colorization/conditioning_images/conditioning_image_3.jpg"],
            ["a man in a black suit", "monochrome", "/home/omen/Desktop/colorization/conditioning_images/conditioning_image_4.jpg"],
            ["the forbidden city in beijing at sunset with a reflection in the water", "monochrome", "/home/omen/Desktop/colorization/conditioning_images/conditioning_image_5.jpg"],
            ["a man in a white shirt holding his hand out in front of", "monochrome", "/home/omen/Desktop/colorization/conditioning_images/conditioning_image_6.jpg"],
        ],
        inputs=[
            prompt, negative_prompt, conditioning_image
        ],
        outputs=output,
        fn=infer,
        cache_examples=True,
    )
    # gr.Markdown(
    #     """
    # * [Dataset](https://huggingface.co/datasets/ioclab/grayscale_image_aesthetic_3M)
    # * [Diffusers model](https://huggingface.co/ioclab/control_v1p_sd15_brightness), [Web UI model](https://huggingface.co/ioclab/ioc-controlnet)
    # * [Training Report](https://api.wandb.ai/links/ciaochaos/oot5cui2), [Doc(Chinese)](https://aigc.ioclab.com/sd-showcase/brightness-controlnet.html)
    # """)

demo.launch(share=True)
