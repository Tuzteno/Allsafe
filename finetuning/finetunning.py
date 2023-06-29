import deepface

# Load the DeepFace library

face_recognition = deepface.DeepFace()

# Set the folder where the facial images are stored

faces_folder = "/app/Fine-Tunning/video-to-frames/images/"

# Download the pre-trained model

vggface2_model = face_recognition.download_model("vggface2")

# Create a training dataset

training_dataset = face_recognition.create_dataset(faces_folder)

# Prepare the training dataset

face_recognition.prepare_dataset(training_dataset)

# Fine-tune the model

face_recognition.fine_tune(training_dataset, vggface2_model)

# Save the model

face_recognition.save_model("/app/model_output")
