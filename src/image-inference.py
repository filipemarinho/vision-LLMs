import os
import PIL.Image

from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree_predictor import (
    TreePredictor, Tree
)
from nanoowl.tree_drawing import (
    draw_tree_output
)
def initialize_predictor(model = "google/owlvit-base-patch32", encoder = "/data/owl_image_encoder_patch32.engine"):
    
    # assert os.path.isfile(model), f"Model {model} does not exist"
    assert os.path.isfile(encoder), f"Encoder {encoder} does not exist"

    predictor = TreePredictor(
        owl_predictor=OwlPredictor(
            model,
            image_encoder_engine=encoder
        )
    )
    print(f"Image Tree Predictor Initialized: \n From model:{model}, encoder: {encoder}")
    return predictor

def tree_infer_image(predictor, image, prompt:str, save_path:str="output.png"):
    image = PIL.Image.open(image)
    tree = Tree.from_prompt(prompt)
    clip_text_encodings = predictor.encode_clip_text(tree)
    owl_text_encodings = predictor.encode_owl_text(tree)

    output = predictor.predict(
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
    predictor = initialize_predictor()
    print("Predictor initialized \n")

    #example from COCO
    prompt = "[a person, a bucket]"
    tree_infer_image(predictor, "/data/sample.jpg", prompt, "/data/output-person-bucket.jpg")
