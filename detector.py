from pathlib import Path  # Helps with file and directory paths
import face_recognition  # Helps with face recognition
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
from PIL import Image, ImageDraw
from collections import Counter

# Setting Default Path
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

# Create an ArgumentParser object with a description
parser = argparse.ArgumentParser(description="Recognize faces in an image")

# Add command-line arguments using add_argument method

# --train argument: a flag to indicate whether to train on input data
parser.add_argument("--train", action="store_true", help="Train on input data")

# --validate argument: a flag to indicate whether to validate trained model
parser.add_argument("--validate", action="store_true", help="Validate trained model")

# --test argument: a flag to indicate whether to test the model with an unknown image
parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")

# -m argument: to specify which model to use for training, default is "hog", choices are "hog" or "cnn"
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)

# -f argument: to specify the path to an image with an unknown face
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)

# Parse the command-line arguments
args = parser.parse_args()

# Creating Folders
Path("training").mkdir(exist_ok=True)  # Creates a folder named "training" if it doesn't exist
Path("output").mkdir(exist_ok=True)     # Creates a folder named "output" if it doesn't exist
Path("validation").mkdir(exist_ok=True) # Creates a folder named "validation" if it doesn't exist

def encode_known_faces(model="hog", encodings_location=DEFAULT_ENCODINGS_PATH):
    """
    This function helps with encoding known faces.
    It loads images of faces from the 'training' directory and encodes them using face recognition.

    RETURNS: no value
    """
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)
        
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    # create a dictionary 
    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
    else:
        return "Unknown"  # Assign "Unknown" label when face cannot be recognized
    
def recognize_faces(image_location, model="hog", encodings_location=DEFAULT_ENCODINGS_PATH):
    """
    Recognizes faces in the given image and returns a list of predicted labels.
    """
    try:
        with encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)
    except FileNotFoundError:
        print("Error: Encodings file not found.")
        return ["Unknown"]  # Return "Unknown" if encodings file not found
    except Exception as e:
        print("Error loading encodings:", e)
        return ["Unknown"]

    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    predicted_labels = []  # List to store predicted labels

    if not input_face_locations:
        # If no faces are detected, return a list with one "Unknown" label
        return ["Unknown"]

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        predicted_labels.append(name)  # Append predicted label
        _display_face(draw, bounding_box, name)
    
    del draw
    pillow_image.show()

    return predicted_labels

BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
   
    draw.rectangle(((left,top), (right,bottom)), outline = BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    
    draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill="blue", outline="blue",)
    
    draw.text((text_left, text_top), name, fill="white",)

def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )

# Define a function to validate and construct the confusion matrix
def get_true_label_from_filepath(filepath):
    # Assuming the file name contains the ground truth label
    true_label = filepath.parent.name  # Assuming the parent directory name is the label
    return true_label            

#In order to get a better Confusion Matrix we need to fine-tune it (e.g changing parameters, normalziing data, rotating the image)
def validate_with_confusion_matrix(model='hog'):
    true_labels = []  # Ground truth labels
    predicted_labels = []  # Predicted labels

    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            image_location = str(filepath.absolute())
            true_label = get_true_label_from_filepath(filepath)
            true_labels.append(true_label)

            # Recognize faces in the image
            predicted_label = recognize_faces(image_location=image_location, model=model)
            predicted_labels.extend(predicted_label)  # Extend the list of predicted labels

    # Ensure consistent label encoding
    label_mapping = {label: idx for idx, label in enumerate(set(true_labels + predicted_labels))}
    true_labels_encoded = [label_mapping[label] for label in true_labels]
    predicted_labels_encoded = [label_mapping[label] for label in predicted_labels]

    # Construct the confusion matrix
    confusion_matrix_array = confusion_matrix(true_labels_encoded, predicted_labels_encoded)
    print("Confusion Matrix:")
    print(confusion_matrix_array)

if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
