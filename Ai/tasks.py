from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from Ai.utils import *
from Ai.parallel_handling import update_progress, verify_for_cancellation

'''
_Outpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(DEFAULT_DEVICE)
'''

_Inpaint_pipeline = pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(DEFAULT_DEVICE)

'''
_Inpaint_pipeline = pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    #"SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
    custom_pipeline="pipeline_stable_diffusion_xl_differential_img2img",
).to(DEFAULT_DEVICE)
'''
_Outpaint_Pipeline = _Inpaint_pipeline

def outpaint_stable_diffusion(task_id, image: Image.Image, model):

    try:
        total_number_of_steps = 30
        strength = 0.8
        def update_the_client(_, step: int, timestep: int, callback_kwargs: dict):
            if verify_for_cancellation(task_id):
                return
            print("Am intrat in callback")
            task = task_id
            total = total_number_of_steps * strength
            percent = int((step / total) * 100)
            print(f"Procent: {percent}")
            try:
                update_progress(task, percent, status='Outpainting')
            except Exception:
                pass
            return {}
        pipeline = _Outpaint_Pipeline
        print("Am intrat in functia de outpaint!")

        original_image = image
        image = prepare_image_for_outpaint(image)
        device = DEFAULT_DEVICE
        img_tensor = transform_image_to_tensor(image)
        img_tensor = img_tensor.to(device)

        print("Am pregatit imaginea")

        try:
            update_progress(task_id, 50, "Generating initial margins...")
            print("Am updatat progress")
        except Exception:
            pass

        if verify_for_cancellation(task_id):
            return

        model.eval()
        with torch.no_grad():
            output = model(img_tensor)  # shape = (1, C_out, H_out, W_out)

        output_image = transform_tensor_to_cv2_image(output)

        try:
            update_progress(task_id, 100, "Generating initial margins...")
        except Exception:
            pass

        new_sizes = int(original_image.width * 256 / 192), int(original_image.height * 256 / 192)
        mask = get_outpaint_mask()
        big_mask = resize_cv_image(mask, new_sizes)
        output_image = resize_cv_image(output_image, new_sizes)
        output_image = cv.GaussianBlur(output_image, (9, 9), 0)

        x, y, _ = np.argwhere(big_mask == 1)[0]
        sizes = original_image.height, original_image.width

        output_image[x:x+sizes[0], y:y+sizes[1], :] = np.array(original_image)[:, :, 0:3]

        big_mask = (1 - big_mask) * 255
        #big_mask = cv.GaussianBlur(big_mask, (9, 9), 0)

        big_mask = Image.fromarray(big_mask)
        output_image = Image.fromarray(output_image)
        if verify_for_cancellation(task_id):
            return

        print('AM dat la pipeline')

        image = pipeline(
            prompt='',
            negative_prompt='',
            num_inference_steps=total_number_of_steps,
            image=output_image,
            strength=strength,
            mask_image=big_mask,
            guidance_scale=0,
            callback_on_step_end=update_the_client
        ).images[0]

        return image
    except Exception as e:
        print(e)

    return None

def inpaint_stable_diffusion(task_id, image: Image.Image, coordinates, model):
    try:
        total_number_of_steps = 40
        strength = 0.4
        def update_the_client(_, step: int, timestep: int, callback_kwargs: dict):
            if verify_for_cancellation(task_id):
                return
            print("Am intrat in callback")
            task = task_id
            total = total_number_of_steps * strength
            percent = int((step / total) * 100)
            print(f"Procent: {percent}")
            try:
                update_progress(task, percent, status='Inpainting')
            except Exception:
                pass
            return {}

        print("Am intrat in functia de inpaint!")

        pipeline = _Inpaint_pipeline

        original_image = image
        image, mask = prepare_image_for_inpaint(image, coordinates)
        device = DEFAULT_DEVICE

        small_mask, small_image = resize_cv_image(mask, (256, 256)), resize_cv_image(image, (256, 256))
        small_mask = 255 - small_mask
        show_image(small_mask)
        img_tensor = transform_image_to_tensor(small_image)
        mask_tensor = transform_image_to_tensor(small_mask)
        mask_tensor = mask_tensor[:, 0:1, :, :]
        img_tensor = torch.cat([img_tensor, mask_tensor], dim=1)
        img_tensor = img_tensor.to(device)

        print("Am pregatit imaginea")

        try:
            update_progress(task_id, 50, "Generating initial margins...")
            print("Am updatat progress")
        except Exception:
            pass

        if verify_for_cancellation(task_id):
            return

        model.eval()
        with torch.no_grad():
            output = model(img_tensor)  # shape = (1, C_out, H_out, W_out)

        output_image = transform_tensor_to_cv2_image(output)

        try:
            update_progress(task_id, 100, "Generating initial margins...")
        except Exception:
            pass


        output_image = cv.resize(output_image, (image.shape[1], image.shape[0]))
        mask = (mask / 255).astype(np.uint8)
        image = (image * mask + output_image * (1 - mask)).astype(np.uint8)

        mask = ((1 - mask) * 255).astype(np.uint8)
        show_image(image)

        image = Image.fromarray(image)
        height, width = image.height, image.width
        mask = Image.fromarray(mask)
        if verify_for_cancellation(task_id):
            return
        print('AM dat la pipeline')

        image = pipeline(
            prompt='',
            negative_prompt='',
            num_inference_steps=total_number_of_steps,
            image=image,
            strength=strength,
            mask_image=mask,
            guidance_scale=0,
            callback_on_step_end=update_the_client
        ).images[0]


        resized = image.resize((width, height), resample=Image.Resampling.LANCZOS)

        return resized

    except Exception as e:
        print(e)

    return None