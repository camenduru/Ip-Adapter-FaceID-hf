import torch
import spaces
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDPlus
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import gradio as gr
import cv2

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sd15.bin", repo_type="model")
ip_plus_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid-plusv2_sd15.bin", repo_type="model")

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
)

ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)
ip_model_plus = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_plus_ckpt, device)

@spaces.GPU(enable_queue=True)
def generate_image(images, prompt, negative_prompt, preserve_face_structure, progress=gr.Progress(track_tqdm=True)):
    pipe.to(device)
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    faceid_all_embeds = []
    first_iteration = True
    for image in images:
        face = cv2.imread(image)
        faces = app.get(face)
        faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        faceid_all_embeds.append(faceid_embed)
        if(first_iteration and preserve_face_structure):
            face_image = face_align.norm_crop(face, landmark=faces[0].kps, image_size=224) # you can also segment the face
            first_iteration = False
            
    average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)
    
    if(not preserve_face_structure):
        print("Generating normal")
        image = ip_model.generate(
            prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=average_embedding,
            width=512, height=512, num_inference_steps=30
        )
    else:
        print("Generating plus")
        image = ip_model_plus.generate(
            prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=average_embedding,
            face_image=face_image, shortcut=True, s_scale=1.5, width=512, height=512, num_inference_steps=30
        )
    print(image)
    return image
css = '''
h1{margin-bottom: 0 !important}
'''
demo = gr.Interface(
        css=css,
        fn=generate_image,
        inputs=[
            gr.Files(
                label="Drag 1 or more photos of your face",
                file_types=["image"]
            ),
            gr.Textbox(label="Prompt",
                       info="Try something like 'a photo of a man/woman/person'",
                       placeholder="A photo of a [man/woman/person]..."),
            gr.Textbox(label="Negative Prompt", placeholder="low quality"),
            gr.Checkbox(label="Preserve Face Structure", info="Higher quality, less versatility (the face structure of your first photo will be preserved)", value=True),
        ],
        outputs=[gr.Gallery(label="Generated Image")],
        title="IP-Adapter-FaceID demo",
        description="Demo for the [h94/IP-Adapter-FaceID model](https://huggingface.co/h94/IP-Adapter-FaceID) - 'preserve face structure' uses the plus v2 model. Non-commercial license",
        allow_flagging=False,
        )
demo.launch()