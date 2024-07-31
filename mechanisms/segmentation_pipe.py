from groundingdino.datasets import transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from torchvision.ops import box_convert
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image, ImageDraw, ImageFont

def grounded_segmentation(gd_model, sam_model, text_prompt, point_coords, point_labels, original_image, box_threshold, text_threshold):

    #Ground
    image_with_box, size, boxes_filt, pred_phrases, pred_dict = ground_image(gd_model, text_prompt, original_image, box_threshold, text_threshold)

    #Seg
    all_masks = sam_seg_rects(sam_model, point_coords, point_labels, original_image, pred_dict["boxes"])

    return image_with_box, size, boxes_filt, pred_phrases, all_masks

def ground_image(gd_model, text_prompt, image, box_threshold, text_threshold):
    image_pil, image = prepare_image(image)
    boxes_filt, pred_phrases = get_grounding_output(
        gd_model, image, text_prompt, box_threshold, text_threshold, token_spans=None
    )

    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }

    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    return image_with_box, size, boxes_filt, pred_phrases, pred_dict

def sam_seg_rects(sam_model, point_coords, point_labels, image, boxes):
    image_pil, image = prepare_image(image)
    cv2_img = set_prediction_target(sam_model, image_pil)

    mask, rects = get_grounding_masks(cv2_img, boxes)

    all_masks = []
    for rect in rects:
        masks, _, _ = sam_model.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=np.array(rect)[None, :],
            multimask_output=False,
        )
        all_masks.append(masks)

    return all_masks

def get_font(font_size):
    # List of common fonts to try
    font_names = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "Helvetica.ttf",
        "Verdana.ttf",
        "FreeSans.ttf",  # Common on Linux
    ]
    
    for font_name in font_names:
        try:
            return ImageFont.truetype(font=font_name, size=font_size)
        except IOError:
            continue
    
    return ImageFont.load_default()

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    font_size = 36

    font = get_font(font_size)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())

        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)

        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white", font=font)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def prepare_image(image):
    image_pil = image.convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    return prepare_image(image_pil)

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


def load_sam_model(model_path):
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

def set_prediction_target(model, image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    model.set_image(image)
    return image

def get_grounding_masks(image, masks):
    h, w, _ = image.shape
    boxes_unnorm = masks * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    mask = np.zeros_like(image)

    input_points = []
    rects = []
    labels = []
    for box in boxes_xyxy:
        x0, y0, x1, y1 = box
        mask[int(y0):int(y1), int(x0):int(x1), :] = 255
        input_points.append([x0, y0])
        input_points.append([x1, y1])
        rects.append([x0, y0, x1, y1])
        labels.extend([1, 0])

    final_points = np.array(input_points)
    return mask, rects

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   