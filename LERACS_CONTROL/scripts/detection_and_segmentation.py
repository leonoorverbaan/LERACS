import cv2
import numpy as np
import json
import re
from typing import List
import os
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class ObjectDetectionAndSegmentation:
    def __init__(self, json_file_path: str, source_image_path: str):
        self.json_file_path = json_file_path
        self.source_image_path = source_image_path
        self.classes = self.load_json_data()
        self.image = cv2.imread(self.source_image_path)
        self.model = self.setup_groundingdino_model()
        self.sam_predictor = SamPredictor(sam_model_registry["vit_h"](
            "/home/leonoor/MRIRAC_Leonoor/src/segment-anything/weights/sam_vit_h_4b8939.pth"))

    def load_json_data(self) -> List[str]:
        with open(self.json_file_path, 'r') as file:
            data = json.load(file)
        object_names = data['environment_before']['objects']
        # Use regex to extract only the base class names without IDs or other annotations
        cleaned_names = [re.sub(r'\(.*\)', '', name).strip() for name in object_names]
        cleaned_names = [re.sub(r'[_]', ' ', name).strip() for name in cleaned_names]
        cleaned_names = [name.replace('<', '').replace('>', '') for name in cleaned_names]
        return cleaned_names

    def setup_groundingdino_model(self) -> Model:
        GROUNDING_DINO_CONFIG_PATH = "/home/leonoor/MRIRAC_Leonoor/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT_PATH = "/home/leonoor/MRIRAC_Leonoor/src/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        return Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    def enhance_class_name(self, class_names: List[str]) -> List[str]:
        return [f"all {class_name}s" for class_name in class_names]

    def detect_objects(self, box_threshold: float, text_threshold: float):
        detections = self.model.predict_with_classes(
            image=self.image,
            classes=self.enhance_class_name(self.classes),
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # Filter out large boxes (example: boxes with width or height > certain value)
        max_width = 400  # Maximum width of the bounding box
        max_height = 400  # Maximum height of the bounding box
        filtered_boxes = [box for box in detections.xyxy if
                          (box[2] - box[0] <= max_width and box[3] - box[1] <= max_height)]
        detections.xyxy = np.array(filtered_boxes)

        return detections

    def segment_objects(self, detections):
        self.sam_predictor.set_image(self.image)  # Initial setting
        result_masks = []
        for box in detections.xyxy:
            self.sam_predictor.set_image(self.image)  # Essential to set the image here again for each detection
            masks, scores, _ = self.sam_predictor.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def annotate_image(self, detections) -> np.ndarray:
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = []
        for _, _, confidence, class_id, _ in detections:
            if class_id is not None:
                label = f"{self.classes[class_id]} {confidence:.2f}"
            else:
                label = f"Unknown {confidence:.2f}"
            labels.append(label)
        annotated_image = mask_annotator.annotate(scene=self.image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image

    def process(self):
        print("Updated CLASSES:", self.classes)
        detections = self.detect_objects(box_threshold=0.28, text_threshold=0.16)
        print("Detections:", detections)
        detections.mask = self.segment_objects(detections)
        annotated_image = self.annotate_image(detections)
        
        # Save the image with a timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"c_scenario12_trial4_seg.jpg"
        output_directory = "/home/leonoor/MRIRAC_Leonoor/src/rcdt_LLM_fr3/MRIRAC_experiment/Data/System/Snapshots"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_filepath = os.path.join(output_directory, output_filename)
        
        cv2.imwrite(output_filepath, annotated_image)
        print(f"Saved segmented and annotated image to {output_filepath}")


if __name__ == "__main__":
    json_file_path = "/home/leonoor/MRIRAC_Leonoor/src/rcdt_LLM_fr3/MRIRAC_experiment/Data/System/Franka_output/c_scenario12_trial4.json"
    source_image_path = "/home/leonoor/MRIRAC_Leonoor/src/rcdt_LLM_fr3/MRIRAC_experiment/Data/System/Snapshots/c_scenario12_trial4.jpg"

    odas = ObjectDetectionAndSegmentation(json_file_path, source_image_path)
    odas.process()
