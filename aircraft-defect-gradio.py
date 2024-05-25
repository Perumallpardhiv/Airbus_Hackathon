import gradio as gr
import PIL.Image as Image
from ultralytics import ASSETS, YOLO

model = YOLO("best-aircraft.pt")

def predict_image(img):
    """Predicts and plots labeled objects in an image using YOLOv8 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=0.25,
        iou=0.45,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        # gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        # gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Ultralytics Gradio",
    description="Upload images for inference. The Ultralytics YOLOv8n model is used by default.",
    examples=[
        ["aircraft 1.jpg"],
        ["aircraft 2.jpg"],
    ],
)

if __name__ == "__main__":
    iface.launch(share=True)
