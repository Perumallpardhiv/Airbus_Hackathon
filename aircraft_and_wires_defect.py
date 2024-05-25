import gradio as gr
import PIL.Image as Image
from ultralytics import YOLO

def predict_image(img, model_type):
    """Predicts and plots labeled objects in an image using YOLOv8 model with adjustable confidence and IOU thresholds."""
    
    if model_type == "Aircraft":
        model = YOLO("best-aircraft.pt")
    elif model_type == "Wires":
        model = YOLO("best-wires.pt") 
    
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
        gr.Dropdown(choices=["Aircraft", "Wires"], label="Choose a model", value="Aircraft"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="<div class='title'>E.D.I.T.H</div>",
    description="<div class='description'>The Risks of Faulty Wiring and Structural Damage in Aircraft</div>",
    examples=[
        ["aircraft 1.jpg", "Aircraft"],
        ["aircraft 2.jpg", "Aircraft"],
        ["wires 1.jpg", "Wires"],
        ["wires 2.jpg", "Wires"],
    ],
    css="style.css"
)

if __name__ == "__main__":
    iface.launch(share=True)
    # iface.launch()
