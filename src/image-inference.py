import os
import PIL.Image
from typing import Optional, Tuple

from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree_predictor import (
    TreePredictor, Tree
)
from nanoowl.tree_drawing import (
    draw_tree_output
)

class ImageInferencePipeline:

    DEFAULT_MODEL = "google/owlvit-base-patch32"
    DEFAULT_ENCODER_PATH = "/data/owl_image_encoder_patch32.engine"

    def __init__(self, model: Optional[str] = None, encoder_path: Optional[str] = None):
        """
        Initialize the image inference pipeline with model and encoder paths.
        """
        self.model = model or self.DEFAULT_MODEL
        self.encoder_path = encoder_path or self.DEFAULT_ENCODER_PATH
        self.predictor = self._initialize_predictor()
    def _initialize_predictor(self) -> TreePredictor:        
        assert os.path.isfile(self.encoder_path), f"Encoder {self.encoder_path} does not exist" #TODO: treat exceptions: if and raise file not found

        predictor = TreePredictor(
            owl_predictor=OwlPredictor(
                self.model,
                image_encoder_engine=self.encoder_path
            )
        )
        print(f"Image Tree Predictor Initialized: \n From model:{self.model}, encoder: {self.encoder_path}")
        return predictor

    def tree_infer_image(self, image, prompt:str, save_path:str="output.png"):
        image = PIL.Image.open(image)
        tree = Tree.from_prompt(prompt)
        clip_text_encodings = self.predictor.encode_clip_text(tree)
        owl_text_encodings = self.predictor.encode_owl_text(tree)

        output = self.predictor.predict(
            image=image, 
            tree=tree,
            clip_text_encodings=clip_text_encodings,
            owl_text_encodings=owl_text_encodings,
            threshold=0.3
        )

        image = draw_tree_output(image, output, tree=tree, draw_text=True)

        image.save(save_path)
        return

if __name__ == "__main__":
    print("Starting... \n")
    predictor = ImageInferencePipeline()
    print("Predictor initialized \n")

    #example from COCO
    prompt = "[a person, a bucket]"
    predictor.tree_infer_image("/data/sample.jpg", prompt, "/data/output-person-bucket.jpg")
