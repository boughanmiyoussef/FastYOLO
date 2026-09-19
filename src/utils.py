

import cv2
import numpy as np

def rel(path, root):
    """
    Get the relative path of a file with respect to a root directory.

    Args:
        path (str): The absolute path of the file.
        root (str): The root directory to which the relative path is calculated.

    Returns:
        str: The relative path of the file with respect to the root directory.
    """
    try:
        return path.relative_to(root)
    except ValueError:
        return path
    
    
def load_class_names(path):
    """"
    Read coco.names into a list, where each line is a class name.
    """
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]
    
    
def output_layer_names(net):
    """
    The layers OpenCV reads the detection results from are the output layers. This function returns the names of those layers.
    """
    layer_names = net.getLayerNames()
    return [layer_names[i -1] for i in net.getUnconnectedOutLayers().flatten()]


def postprocess(outputs, input_size, scale, pad_w, pad_h, conf_thresh, nms_thresh):
    """Turn the raw network output into usable detections. Each candidate keeps
    its best class and score, weak ones are dropped, the rest are mapped from the
    padded letterbox back to original pixels, and overlapping boxes are thinned
    out with non-max suppression. Returns (class_id, confidence, [x, y, w, h]).
    """
    boxes, confidences, class_ids = [], [], []
    for out in outputs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            confidence = float(det[4] * scores[class_id])
            if confidence < conf_thresh:
                continue
            cx = (det[0] * input_size - pad_w) / scale
            cy = (det[1] * input_size - pad_h) / scale
            bw = det[2] * input_size / scale
            bh = det[3] * input_size / scale
            boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
            confidences.append(confidence)
            class_ids.append(class_id)

    detections = []
    if boxes:
        keep = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
        for i in np.array(keep).flatten():
            detections.append((class_ids[i], confidences[i], boxes[i]))
    return detections

def _class_color(class_id):
    """A fixed colour per class, so the same class always draws the same hue."""
    hsv = np.uint8([[[(class_id * 33) % 180, 200, 230]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)

def draw_detections(image, detections, class_names):
    """
    Draw bounding boxes and labels for each detection on the image.
    """
def draw_detections(image, detections, class_names):
    """Draw each box and its label onto a copy of the frame and return it."""
    out = image.copy()
    for class_id, confidence, (x, y, w, h) in detections:
        color = _class_color(class_id)
        label = f"{class_names[class_id]} {confidence:.2f}"
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 2, y), color, -1)
        cv2.putText(out, label, (x + 1, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


class FpsMeter:
    """Keeps a smoothed frames-per-second so the on-screen number doesn't jump
    around on every frame. Each new frame time nudges the running average."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.fps = None

    def update(self, dt):
        inst = 1.0 / dt if dt > 0 else 0.0
        self.fps = inst if self.fps is None else (1 - self.alpha) * self.fps + self.alpha * inst
        return self.fps