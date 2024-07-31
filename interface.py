import gradio as gr
import os
import numpy as np
from PIL import Image
from mechanisms.segmentation_pipe import load_model, load_sam_model, ground_image, sam_seg_rects

CURRENTLY_POSITIVE = True

def get_select_index(evt: gr.SelectData, image, point_label_state):
    global CURRENTLY_POSITIVE
    star_image = Image.open("./star.png") if CURRENTLY_POSITIVE else Image.open("./red_star.png")
    star_image = star_image.resize((32, 32))

    x = evt.index[0]
    y = evt.index[1]
    star_size = 32
    image_width, image_height = star_image.size
    star_x = x - int(image_width // 2)
    star_y = y - int(image_height // 2)

    image.paste(star_image, (star_x, star_y), mask=star_image)
    point_label_state["points"].append((x, y))
    point_label_state["labels"].append(1 if CURRENTLY_POSITIVE else 0)
    return image, point_label_state

def box_segment(image, text_prompt, box_threshold, text_threshold):
    print(image)

    config_file = "./gd_configs/grounding_dino_config.py"
    checkpoint_path = "./checkpoints/groundingdino_swint_ogc.pth"

    model = load_model(config_file, checkpoint_path).eval().to("cuda")
    token_spans = None

    image = image.convert("RGB")
    image_with_box, size, boxes_filt, pred_phrases, pred_dict = ground_image(model, text_prompt, image, box_threshold, text_threshold)

    return image_with_box, pred_dict

def create_mask_and_cutout(image, mask, color=(255, 255, 255)):
    h, w = mask.shape[-2:]
    
    # Create the mask image
    mask_image = Image.fromarray(np.uint8(mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)))
    
    # Create the grayscale mask
    mask_gray = Image.fromarray(np.uint8(mask.reshape(h, w) * 255), mode='L')
    
    # Create the masked cutout image
    masked_image = Image.fromarray(np.array(image))
    masked_image_np = np.array(masked_image)
    masked_image_np[mask.reshape(h, w) == 0] = 0
    masked_cutout = Image.fromarray(masked_image_np)
    
    return mask_gray, masked_cutout

def segment(image, pred_dict, point_label_state):
    boxes = pred_dict["boxes"]

    print(point_label_state)
    print(point_label_state["points"])
    print(point_label_state["labels"])
    if point_label_state["points"] != []:
        points = np.array(point_label_state["points"])
        labels = np.array(point_label_state["labels"])
    else: 
        points = None
        labels = None

    predictor = load_sam_model("./checkpoints/sam2_hiera_large.pt")
    all_masks = sam_seg_rects(
        predictor, 
        points,#np.array([point_coords]), 
        labels, #np.array([1]), 
        image, 
        boxes)

    output_masks = []
    output_cutouts = []
    for mask in all_masks:
        masked_item, cutout = create_mask_and_cutout(image, mask)
        output_masks.append(masked_item)
        output_cutouts.append(cutout)
    return output_cutouts

def passthrough(image):
    return image, {"points":[], "labels":[]}

def toggle_current_pos():
    global CURRENTLY_POSITIVE
    CURRENTLY_POSITIVE = not CURRENTLY_POSITIVE

with gr.Blocks() as demo:
    with gr.Column():
        prompt = gr.Textbox("Prompt")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    box_thresh = gr.Slider(minimum=0.0, value=0.3, maximum=1.0, label="Box Threshold")
                    text_thresh = gr.Slider(minimum=0.0, value=0.3, maximum=1.0, label="Text Threshold")
                ground_button = gr.Button("Ground Image")
            with gr.Column():
                toggle_pos = gr.Button("Toggle Positive")
                segment_button = gr.Button("Segment Image")
    with gr.Row():
        with gr.Column():
            primary_image = gr.Image(
                type="pil", interactive=True,
            )

        with gr.Column():
            box_image = gr.Image(
                type="pil", interactive=False,
            )

        with gr.Column():
            selection_image = gr.Image(
                type="pil", interactive=False,
            )

        with gr.Column():
            final_image = gr.Gallery(
                type="pil", interactive=False,
            )

    toggle_pos.click(toggle_current_pos)

    point_label_state = gr.State(value={"points":[], "labels":[]})
    box_image.change(passthrough, box_image, [selection_image, point_label_state])
    
    selection_image.select(get_select_index, [selection_image, point_label_state], [selection_image, point_label_state])

    pred_state = gr.State()
    ground_button.click(box_segment, [primary_image, prompt, box_thresh, text_thresh], [box_image,pred_state])

    segment_button.click(segment, [primary_image, pred_state, point_label_state], final_image)

demo.launch()