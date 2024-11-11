import os
from PIL import Image
from typing import Optional, Tuple

from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree_predictor import TreePredictor, Tree
from nanoowl.tree_drawing import draw_tree_output

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
        if not os.path.isfile(self.encoder_path):
            raise FileNotFoundError(f"Encoder file does not exist: {self.encoder_path}")
        
        owl_predictor = OwlPredictor(
                self.model,
                image_encoder_engine=self.encoder_path
        )

        predictor = TreePredictor(owl_predictor)

        print(f"Image Tree Predictor Initialized: \n From model:{self.model}, encoder: {self.encoder_path}")
        return predictor

    def load_image(self, image_path: str) -> Image.Image:
        """
        Loads an image from the specified path.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        return Image.open(image_path)

    def save_image(self, image: Image.Image, save_path: str) -> None:
        """
        Saves the processed image to a specified path.
        """
        image.save(save_path)
        print(f"Image saved to: {save_path}")

    def tree_infer_image(self, image: Image.Image, prompt: str, threshold: float = 0.3) -> Image.Image:
        """
        Runs the prediction on the image using the predictor and tree structure.
        """      
        # Extract detection tree from prompt
        tree = Tree.from_prompt(prompt)

        # Encode text embeddings
        clip_text_encodings = self.predictor.encode_clip_text(tree)
        owl_text_encodings = self.predictor.encode_owl_text(tree)

        output = self.predictor.predict(
            image=image, 
            tree=tree,
            clip_text_encodings=clip_text_encodings,
            owl_text_encodings=owl_text_encodings,
            threshold=threshold
        )

        image = draw_tree_output(image, output, tree=tree, draw_text=True)

        return image
    
    def process_inference(self, image_path: str, prompt: str, save_path: Optional[str] = None, threshold: float = 0.3) -> None:
        """
        Full inference pipeline: loads image, prepares tree, encodes, infers, and saves output.
        """
        # Load image
        image = self.load_image(image_path)

        # Perform inference
        processed_image = self.tree_infer_image(image, prompt, threshold)

        # Save output image
        if save_path: self.save_image(processed_image, save_path)
        return

if __name__ == "__main__":
    print("Starting... \n")
    predictor = ImageInferencePipeline()
    print("Predictor initialized \n")

    #example from COCO
    prompt = "[a person, a bucket]"
    predictor.process_inference("/data/sample.jpg", prompt, "/data/output-person-bucket.jpg")
