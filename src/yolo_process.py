from ultralytics import YOLO


def init_yolo(yolo_model_path):
    try:
        yolo_model = YOLO(yolo_model_path)
        print(f"Successfully loaded YOLO model: {yolo_model_path}")
        return yolo_model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def yolo_predict(yolo_model, pil_image, image_entry: dict, conf_thresh: float):
    try:
        results = yolo_model.predict(source=pil_image, conf=conf_thresh, verbose=False)
        if results and results[0].obb is not None:
            obb_data = results[0].obb
            detected_classes = obb_data.cls.cpu().numpy()
            obb_coords_flat_all = obb_data.xyxyxyxy.cpu().numpy()
            for i in range(len(detected_classes)):
                obb_flat_coords = obb_coords_flat_all[i]
                object_entry = {
                    "id": i,
                    "coords": obb_flat_coords,
                }
                image_entry["objects"].append(object_entry)
    
    except Exception as e:
        print(f"Error processing image in yolo prediction: {e}")
        import traceback
        traceback.print_exc() # This will print the full traceback
        image_entry["error"] = str(e)
    
    return image_entry