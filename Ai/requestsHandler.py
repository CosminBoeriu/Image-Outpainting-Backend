import json
import time

import PIL.Image
from Ai.utils import *
from Ai.Outpaint.outpaintModel import AOTGenerator as OutpaintModel
from Ai.Inpaint.inpaintModel2 import InpaintGAN2 as InpaintModel2
from Ai.parallel_handling import *
from Ai.tasks import outpaint_stable_diffusion, inpaint_stable_diffusion
import io


class RequestsHandler:
    def __init__(self):
        self.outpaint_model = OutpaintModel()
        params = self.outpaint_model.parameters()
        if True:
            params = filter(lambda p: p.requires_grad, params)
        print(f'{sum(p.numel() for p in params)} Outpaint')
        self.outpaint_model.load_state_dict(torch.load('./Outpaint/outpaintModel.pth', map_location=DEFAULT_DEVICE, weights_only=True))
        self.outpaint_model.eval()
        self.outpaint_model.to(DEFAULT_DEVICE)

        self.inpaint_model = InpaintModel2()
        params = self.inpaint_model.parameters()
        if True:
            params = filter(lambda p: p.requires_grad, params)
        print(f'{sum(p.numel() for p in params)} Inpaint')
        self.inpaint_model.load_state_dict(torch.load('./Inpaint/inpaintModel2.pth', map_location=DEFAULT_DEVICE, weights_only=True))
        self.inpaint_model.eval()
        self.inpaint_model.to(DEFAULT_DEVICE)

    def cancel(self, task_id):
        if not isinstance(task_id, int):
            task_id = int(task_id)
        cancel_task(task_id)
        print("Canceled")
        return "Canceled"

    def get_task_result(self, task_id):
        if not isinstance(task_id, int):
            task_id = int(task_id)
        if get_task_status(task_id) != 'DONE':
            return None
        pil_img = get_task_result(task_id)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    def get_request_status(self, task_id):
        if not isinstance(task_id, int):
            task_id = int(task_id)
        status = get_task_status(task_id)
        return status, get_progress(task_id)


    def outpaint(self, image):
        if not isinstance(image, Image.Image):
            pil_img = Image.open(image.stream)
            pil_copy = pil_img.copy()
            pil_img.close()
            image.stream.close()
        else:
            pil_copy = image.copy()
        task_id = submit_task(outpaint_stable_diffusion, pil_copy, self.outpaint_model)
        return task_id

    def inpaint(self, image, mask_box):
        try:
            rect = json.loads(mask_box)
            x1 = rect["x"]
            y1 = rect["y"]
            x2 = x1 + rect["width"]
            y2 = y1 + rect["height"]
        except Exception as e:
            x1, y1, w, h = mask_box
            x2, y2 = x1 + w, y1 + h
        if not isinstance(image, Image.Image):
            pil_img = Image.open(image.stream)
            pil_copy = pil_img.copy()
            pil_img.close()
            image.stream.close()
        else:
            pil_copy = image.copy()
        #task_id = submit_task(inpaint_stable_diffusion, pil_copy, (x1, y1, x2, y2), self.inpaint_model)
        image = inpaint_stable_diffusion(0, pil_copy, (x1, y1, x2, y2), model=self.inpaint_model)
        return image


REQUESTS_HANDLER = RequestsHandler()

if __name__ == '__main__':
    myIMG = PIL.Image.open('./Assets/mountain.jpg')
    REQUESTS_HANDLER.inpaint(myIMG, (0, 250, 300, 500))



