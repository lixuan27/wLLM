from PIL import Image
import numpy as np

def resize_center_crop(img_path, target_size, normalize: bool = False, CHW: bool = False, dtype: np.dtype = np.uint8):
        """
        Read image -> Resize -> Center crop -> Convert to torch.tensor -> Normalize to [-1, 1]
        :param img_path: Path to the image
        :param target_size: (width, height)
        :return: torch.Tensor, shape [C, H, W], range [-1, 1]
        """
        img = Image.open(img_path).convert("RGB")
        target_w, target_h = target_size
        orig_w, orig_h = img.size

        # --- Step 1: proportional resize ---
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # --- Step 2: center crop ---
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        img = img.crop((left, top, right, bottom))

        # --- Step 3: to numpy，and then tensor ---
        img_np = np.array(img).astype(dtype)
        if CHW:
            img_np = img_np.transpose(2, 0, 1)  # [C, H, W], 0~255
        if normalize:
            img_np = img_np.astype(np.float32) / 127.5 - 1.0  # normalize to [-1, 1]
        return img_np
