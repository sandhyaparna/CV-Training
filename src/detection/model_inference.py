from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image


# -----------------------
# Inference helpers
# -----------------------
_to_tensor = T.Compose([T.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def run_inference(model, pil_img, score_thr=0.5):
    img = _to_tensor(pil_img).to(device)
    out = model([img])[0]  # dict: boxes, labels, scores
    keep = out["scores"] >= score_thr
    boxes = out["boxes"][keep].detach().cpu()
    labels = out["labels"][keep].detach().cpu()
    scores = out["scores"][keep].detach().cpu()
    return boxes, labels, scores

def _draw(ax, pil_img, boxes, labels, scores, id2name, title):
    ax.imshow(pil_img)
    ax.set_title(title)
    ax.axis("off")
    for b, l, s in zip(boxes, labels, scores):
        x1, y1, x2, y2 = b.tolist()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
        ax.add_patch(rect)
        cls = id2name.get(int(l), str(int(l)))
        ax.text(
            x1, max(y1 - 3, 0),
            f"{cls} {s:.2f}",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.5, pad=2),
            color="white"
        )

def compare_two_models(img_path, v1_model, v1_model_name, v2_model, v2_model_name, id2name, score_thr=0.5, save_dir=None):
    pil_img = Image.open(img_path).convert("RGB")
    t_b, t_l, t_s = run_inference(v1_model, pil_img, score_thr=score_thr)
    s_b, s_l, s_s = run_inference(v2_model, pil_img, score_thr=score_thr)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    _draw(axes[0], pil_img, t_b, t_l, t_s, id2name, title=f"{v1_model_name}  (n={len(t_b)})")
    _draw(axes[1], pil_img, s_b, s_l, s_s, id2name, title=f"{v2_model_name}  (n={len(s_b)})")
    fig.suptitle(Path(img_path).name, fontsize=12)
    plt.show()


##### VIDEO INFERENCE ######
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import draw_bounding_boxes

# ---- device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_detections(img_pil, model, class_names, score_threshold=0.5, box_width=4):
    """
    img_pil: PIL.Image (RGB)
    class_names: dict mapping class_id -> string, e.g. {1: 'chuck'}
    returns: PIL.Image with boxes drawn
    """
    transform = transforms.ToTensor()
    img_tensor = transform(img_pil).to(device)  # [C,H,W] on same device as model

    model.eval()
    with torch.no_grad():
        outputs = model([img_tensor])  # list of tensors

    result = outputs[0]
    scores = result["scores"]
    keep = scores > score_threshold
    
    boxes = result["boxes"][keep].detach().to("cpu")
    labels = result["labels"][keep].to("cpu")
    kept_scores = scores[keep].detach().to("cpu")

    # prepare label strings
    label_names_raw = [class_names.get(lbl.item(), str(lbl.item())) for lbl in labels]
    label_names = [f"{name} - {score:.2f}" for name, score in zip(label_names_raw, kept_scores)]

    # draw on a CPU tensor version of the image (uint8 expected)
    img_cpu = (img_tensor.to("cpu") * 255).byte()
    # warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
    img_vis = draw_bounding_boxes(
        img_cpu, boxes, labels=label_names, width=box_width, colors="black"
    )

    return torchvision.transforms.ToPILImage()(img_vis)


# # OPTIONAL: stub for display_text_block if you don't already have it
def display_text_block(
    frame_bgr,
    lines,
    x_start=10,
    y_start=None,
    x_gap=0,
    y_gap=20,
    text_color=(255, 255, 255),
    background_color=(0, 0, 0),
):
    # Get frame dimensions
    height, width = frame_bgr.shape[:2]

    # If y_start is not provided, default to near bottom
    if y_start is None:
        y_start = height - 90

    overlay = frame_bgr.copy()
    x, y = x_start, y_start

    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            overlay, (x - 5, y - th - 5), (x + tw + 5, y + 5), background_color, -1
        )
        cv2.putText(
            overlay,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
            cv2.LINE_AA,
        )
        x += x_gap
        y += y_gap

    alpha = 0.6
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)


#### Quant int8 ####
@torch.inference_mode()
def infer_image(model, pil_img, score_thresh=0.5, max_dets=None):
    img_t = to_tensor(pil_img)

    out = model([img_t])

    # TorchScripted torchvision detection models return (losses, detections)
    # in scripting mode. We only need detections for inference.
    if isinstance(out, (tuple, list)) and len(out) == 2:
        _, detections = out
    else:
        detections = out

    det = detections[0]  # first image in the batch

    # Some safety checks
    if not isinstance(det, dict):
        raise RuntimeError(f"Unexpected detection output type: {type(det)}")

    boxes  = det.get("boxes",  torch.empty(0, 4))
    labels = det.get("labels", torch.empty(0, dtype=torch.long))
    scores = det.get("scores", torch.empty(0))

    if boxes.numel() == 0:
        return {"boxes": boxes, "labels": labels, "scores": scores}

    keep = scores >= score_thresh
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    if max_dets is not None and len(scores) > max_dets:
        topk = torch.topk(scores, k=max_dets).indices
        boxes, labels, scores = boxes[topk], labels[topk], scores[topk]

    return {"boxes": boxes, "labels": labels, "scores": scores}


def visualize_and_save(pil_img, det, save_path, id2name=None):
    img_t_uint8 = (to_tensor(pil_img) * 255).to(torch.uint8)
    boxes, labels, scores = det["boxes"], det["labels"], det["scores"]

    if boxes.numel() == 0:
        pil_img.save(save_path)
        return

    if id2name is None: id2name = {}
    labels_txt = [f"{id2name.get(int(l), str(int(l)))}:{float(s):.2f}"
                  for l, s in zip(labels.tolist(), scores.tolist())]

    vis = draw_bounding_boxes(img_t_uint8, boxes, labels=labels_txt, width=2)
    Image.fromarray(vis.permute(1, 2, 0).cpu().numpy()).save(save_path)


##### Quant - fp16 model #####
@torch.inference_mode()
def infer_fp16_image(model, pil_img, score_thresh=0.5, max_dets=None):
    img_t = to_tensor(pil_img).to(device)

    # Use autocast only on CUDA for FP16 math; CPU remains float32
    use_amp = (device.type == "cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        out = model([img_t])

    # TorchScripted torchvision detection models may return (losses, detections)
    # in scripting mode. We only need detections for inference.
    if isinstance(out, (tuple, list)) and len(out) == 2:
        _, detections = out
    else:
        detections = out

    det = detections[0]  # first image

    # Safety checks
    if not isinstance(det, dict):
        raise RuntimeError(f"Unexpected detection output type: {type(det)}")

    boxes  = det.get("boxes",  torch.empty(0, 4, device=device))
    labels = det.get("labels", torch.empty(0, dtype=torch.long, device=device))
    scores = det.get("scores", torch.empty(0, device=device))

    if boxes.numel() == 0:
        return {"boxes": boxes.cpu(), "labels": labels.cpu(), "scores": scores.cpu()}

    keep = scores >= score_thresh
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    if max_dets is not None and len(scores) > max_dets:
        topk = torch.topk(scores, k=max_dets).indices
        boxes, labels, scores = boxes[topk], labels[topk], scores[topk]

    # Move back to CPU for visualization
    return {"boxes": boxes.cpu(), "labels": labels.cpu(), "scores": scores.cpu()}


def visualize_fp16(pil_img, det, id2name=None):
    img_t_uint8 = (to_tensor(pil_img) * 255).to(torch.uint8)
    boxes, labels, scores = det["boxes"], det["labels"], det["scores"]

    if boxes.numel() == 0:
        # pil_img.save(save_path)
        return pil_img

    if id2name is None: id2name = {}
    labels_txt = [f"{id2name.get(int(l), str(int(l)))}:{float(s):.2f}"
                  for l, s in zip(labels.tolist(), scores.tolist())]

    vis = draw_bounding_boxes(img_t_uint8, boxes, labels=labels_txt, colors="black", width=2)
    # Image.fromarray(vis.permute(1, 2, 0).numpy()).save(save_path)
    return torchvision.transforms.ToPILImage()(vis)








