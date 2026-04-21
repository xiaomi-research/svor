import os
import gc
import imageio
import inspect
import numpy as np
import torch
import torchvision
import cv2
from einops import rearrange
from PIL import Image
import time
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def get_width_and_height_from_image_and_base_resolution(image, base_resolution):
    target_pixels = int(base_resolution) * int(base_resolution)
    original_width, original_height = Image.open(image).size
    ratio = (target_pixels / (original_width * original_height)) ** 0.5
    width_slider = round(original_width * ratio)
    height_slider = round(original_height * ratio)
    return height_slider, width_slider


def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst


def save_videos_grid(
    videos: torch.Tensor,
    path: str,
    rescale=False,
    n_rows=6,
    fps=12,
    imageio_backend=True,
    color_transfer_post_process=False,
):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs, fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1 / fps))
    else:
        if path.endswith("mp4"):
            path = path.replace(".mp4", ".gif")
        outputs[0].save(path, format="GIF", append_images=outputs, save_all=True, duration=100, loop=0)


def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [
                    torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
                    for _image_start in image_start
                ],
                dim=2,
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, : len(image_start)] = start_video

            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start) :] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0),
                [1, 1, video_length, 1, 1],
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if type(image_end) is list:
            image_end = [
                _image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
                for _image_end in image_end
            ]
            end_video = torch.cat(
                [
                    torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
                    for _image_end in image_end
                ],
                dim=2,
            )
            input_video[:, :, -len(end_video) :] = end_video

            input_video_mask[:, :, -len(image_end) :] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [
                    torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
                    for _image_start in image_start
                ],
                dim=2,
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, : len(image_start)] = start_video
            input_video = input_video / 255

            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start) :] = 255
        else:
            input_video = (
                torch.tile(
                    torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0),
                    [1, 1, video_length, 1, 1],
                )
                / 255
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[
                :,
                :,
                1:,
            ] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    gc.collect()

    return input_video, input_video_mask, clip_image


def get_video_to_video_latent(
    input_video_path, video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None
):
    if input_video_path is not None:
        if isinstance(input_video_path, str):
            cap = cv2.VideoCapture(input_video_path)
            input_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else int(original_fps // fps)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            input_video = input_video_path

        input_video = torch.from_numpy(np.array(input_video))[:video_length]
        input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

        if validation_video_mask is not None:
            validation_video_mask = (
                Image.open(validation_video_mask).convert("L").resize((sample_size[1], sample_size[0]))
            )
            input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)

            input_video_mask = (
                torch.from_numpy(np.array(input_video_mask))
                .unsqueeze(0)
                .unsqueeze(-1)
                .permute([3, 0, 1, 2])
                .unsqueeze(0)
            )
            input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
            input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
        else:
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, :] = 255
    else:
        input_video, input_video_mask = None, None

    if ref_image is not None:
        if isinstance(ref_image, str):
            clip_image = Image.open(ref_image).convert("RGB")
        else:
            clip_image = Image.fromarray(np.array(ref_image, np.uint8))
    else:
        clip_image = None

    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
    return input_video, input_video_mask, ref_image, clip_image


def process_video(input_video_path, input_mask_video_path, ref_images, video_length, sample_size):
    """Process input video and mask for editing"""
    if input_video_path is not None:
        cap = cv2.VideoCapture(input_video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()

        frames = frames[:video_length]
        if len(frames) < video_length:
            frames += [frames[-1]] * (video_length - len(frames))

        resized_frames = [frame.resize([sample_size[1], sample_size[0]]) for frame in frames]

        input_video = (
            torch.stack([torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in resized_frames])
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
        )  # [1, C, T, H, W]

    else:
        # 直接生成全零张量，形状为 [1, 3, T, H, W]，归一化到 [-1, 1]
        input_video = torch.zeros((1, 3, video_length, sample_size[0], sample_size[1])).float()
    # 生成 input_video_mask 张量
    if input_mask_video_path is not None:
        mask_cap = cv2.VideoCapture(input_mask_video_path)
        mask_frames = []
        while mask_cap.isOpened():
            ret, frame = mask_cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
            mask_frames.append(mask)
        mask_cap.release()

        mask_frames = mask_frames[:video_length]
        if len(mask_frames) < video_length:
            mask_frames += [mask_frames[-1]] * (video_length - len(mask_frames))

        resized_masks = [Image.fromarray(mask).resize([sample_size[1], sample_size[0]]) for mask in mask_frames]

        input_video_mask = (
            torch.stack([torch.from_numpy(np.array(mask)) for mask in resized_masks]).unsqueeze(0).unsqueeze(0) / 255.0
        )  # [1, 1, T, H, W]

    else:
        # 直接生成全1张量，形状为 [1, 1, T, H, W]，表示所有区域有效
        input_video_mask = torch.ones((1, 1, video_length, sample_size[0], sample_size[1])).float()

    if input_video_path is not None and input_video is not None:
        input_video = input_video * (torch.tile(input_video_mask, [1, 3, 1, 1, 1]) < 0.5) + (128.0) * (
            torch.tile(input_video_mask, [1, 3, 1, 1, 1]) >= 0.5
        )
    input_video = input_video.div_(127.5).sub_(1.0)

    if ref_images is not None:
        for i, ref_img in enumerate(ref_images):
            if ref_img is not None:
                ref_img = Image.open(ref_img).convert("RGB")
                ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                if ref_img.shape[-2:] != sample_size:
                    canvas_height, canvas_width = sample_size
                    ref_height, ref_width = ref_img.shape[-2:]
                    white_canvas = torch.ones((3, 1, canvas_height, canvas_width))  # [-1, 1]
                    scale = min(canvas_height / ref_height, canvas_width / ref_width)
                    new_height = int(ref_height * scale)
                    new_width = int(ref_width * scale)
                    resized_image = (
                        F.interpolate(
                            ref_img.squeeze(1).unsqueeze(0),
                            size=(new_height, new_width),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .unsqueeze(1)
                    )
                    top = (canvas_height - new_height) // 2
                    left = (canvas_width - new_width) // 2
                    white_canvas[:, :, top : top + new_height, left : left + new_width] = resized_image
                    ref_img = white_canvas
                ref_images[i] = ref_img
        ref_images = torch.cat(ref_images, dim=1).unsqueeze(0)

    return input_video, input_video_mask, ref_images


class Time_Logger:
    total = 0
    count = 0
    name = None

    def log(self, t):
        self.total += t
        self.count += 1

    def get_avg(self):
        # 转换为秒
        return self.total / self.count


def get_time_stat(logger: Time_Logger):
    def time_stat(func):
        logger.name = func.__name__

        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.log(end - start)
            return result

        return wrapper

    return time_stat


def interpolate_MUSE(masks, new_depth):
    bs, c, depth, height, width = masks.shape
    if depth > 1:
        first_frame = masks[:, :, 0:1, :, :]
        remaining_frames = masks[:, :, 1:, :, :]

        if (depth - 1) % 4 == 0:
            # 重新排列为分组形式以便进行最大池化
            grouped_frames = remaining_frames.view(
                bs,
                c,
                (depth - 1) // 4,  # 组数
                4,  # 每组4帧
                height,
                width,
            )
            # 对每组4帧进行最大池化
            pooled_frames = torch.max(grouped_frames, dim=3)[0]  # 在第4个维度(每组帧)上取最大值
            # 合并首帧和池化后的帧
            masks = torch.cat([first_frame, pooled_frames], dim=2)
        else:
            # 如果不能被4整除，使用原来的插值方法作为后备
            masks = F.interpolate(masks, size=(new_depth, height, width), mode="nearest-exact")
    else:
        masks = masks
    return masks


def check_noise_predictions(noise_pred, step_index, timestep, name="NOISE"):
    """
    检查预测噪声是否有问题
    """
    print(f"\n[{name} CHECK] Step {step_index}, Timestep: {timestep}")

    # 基本形状信息
    print(f"  Noise shape: {noise_pred.shape}")

    # 检查NaN值
    nan_count = torch.isnan(noise_pred).sum().item()
    if nan_count > 0:
        print(f"  *** ERROR: Found {nan_count} NaN values in noise prediction! ***")
        return False

    # 检查无穷大值
    inf_count = torch.isinf(noise_pred).sum().item()
    if inf_count > 0:
        print(f"  *** ERROR: Found {inf_count} Inf values in noise prediction! ***")
        return False

    # 统计信息
    with torch.no_grad():
        mean_val = noise_pred.mean().item()
        std_val = noise_pred.std().item()
        min_val = noise_pred.min().item()
        max_val = noise_pred.max().item()
        abs_max = noise_pred.abs().max().item()

    print(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}")
    print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}")
    print(f"  Abs Max: {abs_max:.6f}")

    # 检查异常大的值
    if abs_max > 1e6:
        print(f"  *** WARNING: Very large values detected (>{1e6})! ***")
        return False

    # 检查零值过多（可能表示数值下溢）
    zero_ratio = (noise_pred.abs() < 1e-10).sum().item() / noise_pred.numel()
    if zero_ratio > 0.5:  # 超过50%的值接近零
        print(f"  *** WARNING: High zero ratio ({zero_ratio * 100:.2f}%) detected! ***")

    # 检查梯度爆炸迹象
    if abs_max > 1e3:
        print(f"  *** WARNING: Large values ({abs_max:.2f}) detected, potential gradient explosion! ***")

    return True


def check_weight_overflow(model, name_prefix=""):
    """检查模型权重是否存在溢出"""
    overflow_detected = False

    for name, param in model.named_parameters():
        full_name = f"{name_prefix}.{name}" if name_prefix else name

        # 计算统计信息
        mean_val = param.mean().item()
        std_val = param.std().item()
        min_val = param.min().item()
        max_val = param.max().item()

        print(f"[STATS] {full_name}: mean={mean_val:.6f}, std={std_val:.6f}, min={min_val:.6f}, max={max_val:.6f}")

        # 检查NaN
        if torch.isnan(param).any():
            print(f"[ERROR] NaN detected in {full_name}")
            overflow_detected = True

        # 检查无穷大值
        if torch.isinf(param).any():
            print(f"[ERROR] Inf detected in {full_name}")
            overflow_detected = True

        # 检查异常大的值
        max_val = param.abs().max().item()
        if max_val > 1e6:
            print(f"[WARN] Large values detected in {full_name}: max={max_val}")

        # 检查零值过多（可能表示数值下溢）
        zero_ratio = (param.abs() < 1e-10).sum().item() / param.numel()
        if zero_ratio > 0.9:
            print(f"[WARN] High zero ratio in {full_name}: {zero_ratio * 100:.2f}%")

    return overflow_detected


def postprocess_videoframe(videos: torch.Tensor, rescale=False, n_rows=6):

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    current_h, current_w = outputs[0].shape[:2]

    pil_outputs = []
    for i, img in enumerate(outputs):
        try:
            # 1. 如果是 PyTorch Tensor，先转到 CPU Numpy
            if hasattr(img, "cpu"):
                img = img.detach().cpu().numpy()

            # 2. 如果是 Numpy 数组
            if isinstance(img, np.ndarray):
                # 检查维度: 如果是 (C, H, W) -> 转为 (H, W, C)
                # 判据: 第一个维度是 3，且最后两个维度比较大
                if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] > 3:
                    img = img.transpose(1, 2, 0)

                # 检查数值范围: 如果是 0.0-1.0 的浮点数 -> 转为 0-255 uint8
                if img.dtype != np.uint8:
                    if img.max() <= 1.05:  # 稍微放宽一点防止 1.0001
                        img = (img * 255).clip(0, 255)
                    img = img.astype(np.uint8)

            # 3. 转换为 PIL
            pil_outputs.append(Image.fromarray(img))

        except Exception as e:
            print(f"[Error] Frame {i} convert failed: {e}. type={type(img)}, shape={getattr(img, 'shape', 'N/A')}")
            # 出错时放入一张黑图防止整个接口崩溃
            pil_outputs.append(Image.new("RGB", (current_w, current_h), (0, 0, 0)))

    return pil_outputs
