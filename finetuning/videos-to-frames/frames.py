import torch
import torchvision.transforms as transforms
from torchvision.io import read_video, write_jpeg

# Load the video
video_path = '/Fine-Tunning/video-to-frames/videos/myself.mp4'
video, audio, info = read_video(video_path)

# Define frame extraction interval
frame_interval = 10  # Extract every 10th frame

# Define transformation for image encoding
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize frames if required
    transforms.ToTensor()
])

# Iterate over frames and extract
frames = []
for frame_idx in range(0, video.shape[0], frame_interval):
    frame = video[frame_idx]
    frame = transform(frame)
    frames.append(frame)

# Convert frames to a tensor
frames_tensor = torch.stack(frames)

# Load a pre-trained model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
model.eval()

# Process frames with the model
with torch.no_grad():
    features = model.extract_features(frames_tensor)

# Save the extracted images
output_path = '/Fine-Tunning/video-to-frames/images/'
for i in range(len(features)):
    image_path = output_path + f'frame_{i}.jpg'
    write_jpeg(frames[i], image_path)