# LogoCraft
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import random
from google.colab import files
from google.colab import drive

def generate_logo(prompt, model_id="CompVis/stable-diffusion-v1-4", device=None, num_images=1, guidance_scale=7.5, num_inference_steps=50):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    images = []

    for _ in range(num_images):
        random_elements = ["modern", "abstract", "geometric", "minimalist", "professional", "creative", "bold", "elegant"]
        augmented_prompt = prompt + " " + random.choice(random_elements)
        image = pipe(augmented_prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        images.append(image)

    return images

def generate_logo_from_image(prompt, input_image_path, model_id="CompVis/stable-diffusion-v1-4", device=None, strength=0.75, guidance_scale=7.5, num_inference_steps=50):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to(device)
    init_image = Image.open(input_image_path).convert("RGB").resize((512, 512))
    random_elements = ["modern", "abstract", "geometric", "minimalist", "professional", "creative", "bold", "elegant"]
    augmented_prompt = prompt + " " + random.choice(random_elements)

    image = pipe(prompt=augmented_prompt, image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
    return image

if __name__ == "__main__":
    option = input("Choose mode (1: Text-to-Logo, 2: Image-to-Logo): ")
    prompt = input("Enter your logo description: ")

    if option == "1":
        generated_logos = generate_logo(prompt, num_images=3)
        for i, logo in enumerate(generated_logos):
            logo.save(f"logo_{i}.png")
            print(f"Logo {i} saved as logo_{i}.png")
    elif option == "2":
        upload_choice = input("Upload image (1) or use Google Drive (2)? ")
        if upload_choice == "1":
            uploaded = files.upload()
            input_image_path = list(uploaded.keys())[0]
        elif upload_choice == "2":
            drive.mount('/content/drive')
            input_image_path = input("Enter full path to image in Google Drive: ")

        generated_logo = generate_logo_from_image(prompt, input_image_path)
        generated_logo.save("new_logo.png")
        generated_logo.show()
        print("New logo saved as new_logo.png")
