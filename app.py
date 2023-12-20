import torch
import spaces
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
import gradio as gr

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_ckpt = hf_hub_download(repo_id='h94/IP-Adapter-FaceID', filename="ip-adapter-faceid_sd15.bin", repo_type="model")

device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    #feature_extractor=None,
    #safety_checker=None
)

ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)

@spaces.GPU
def generate_image(image, prompt, negative_prompt):
    pipe.to(device)
    
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(image)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    
    image = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, width=512, height=512, num_inference_steps=30
    )
    print(image)
    return image

demo = gr.Interface(fn=generate_image,
                    inputs=[gr.Image(label="Your face"),gr.Textbox(label="Prompt"), gr.Textbox(label="Negative Prompt")],
                    outputs=[gr.Gallery(label="Generated Image")],
                    title="IP-Adapter-FaceID demo",
                    description="Demo for the [h94/IP-Adapter-FaceID model](https://huggingface.co/h94/IP-Adapter-FaceID)"
                   )
demo.launch()