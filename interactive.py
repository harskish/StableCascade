import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import random
from pyviewer.single_image_viewer import draw
from pyviewer.utils import reshape_grid

torch.set_grad_enabled(False)

device = "cuda"
B = 1

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device)
prior.safety_checker = None
prior.requires_safety_checker = False

decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch.float16).to(device)
decoder.safety_checker = None
decoder.requires_safety_checker = False

if prior.dtype == torch.bfloat16:
    # cutlassF: no kernel found to launch!
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

prompt = 'A cozy mansion in the Alps'
negative_prompt = "low quality, blurry"
input_seed = random.randint(0, 9999999999)
print('Seed:', input_seed)

generator = torch.Generator(device=device).manual_seed(input_seed)
preview_gen = torch.Generator(device=device).manual_seed(input_seed)

def cbk(self, i, t, kwargs):
    if i % 8 == 0:
        decoder.set_progress_bar_config(disable=True)
        preview = decoder(
            image_embeddings=kwargs['latents'].to(decoder.dtype),
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            generator=preview_gen,
            num_inference_steps=3,
            output_type="pt"
        ).images # CHW
        draw(img_hwc=reshape_grid(preview))

    return kwargs

# Stage C: generate Bx16x24x24 latent
prior_output = prior(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=generator,
    width=1024,
    height=1024,
    guidance_scale=6,
    num_inference_steps=20,
    num_images_per_prompt=B,
    callback_on_step_end_tensor_inputs=['latents'],
    callback_on_step_end=cbk,
)

# Stage B: renife to Bx4x256x256 latent
decoder.set_progress_bar_config(disable=False)
latents = decoder(
    image_embeddings=prior_output.image_embeddings.to(decoder.dtype),
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    generator=generator,
    num_inference_steps=12,
    output_type="latent"
).images

# Stage A: Scale and decode with vq-vae (vq-GAN?)
latents = decoder.vqgan.config.scale_factor * latents
images = decoder.vqgan.decode(latents).sample.clamp(0, 1)

# Profit
draw(img_hwc=reshape_grid(images))

print("Done")
