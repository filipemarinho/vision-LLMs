from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree_predictor import (
    TreePredictor, Tree
)
from nanoowl.tree_drawing import (
    draw_tree_output
)

if __name__ == "__main__":
    predictor = TreePredictor(
        owl_predictor=OwlPredictor(
            "google/owlvit-base-patch32",
            image_encoder_engine="/data/owl_image_encoder_patch32.engine"
        )
    )
    print("Starting...")