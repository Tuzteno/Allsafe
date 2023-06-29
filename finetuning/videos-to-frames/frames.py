import torch
import torchvision.transforms as transforms
from torchvision.io import read_video, write_jpeg
from torchvision.models import resnet50

# Load the video
video_path = '/home/hadmin/Allsafe/finetuning/videos-to-frames/videos/myself.mov'
video, audio, info = read_video(video_path, pts_unit='sec')  # Specify pts_unit as 'sec' for .mov format

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
    if frame.shape[0] > 4:
        frame = frame[:3]  # Keep only the first 3 channels (RGB)
    frame = transform(frame)
    frames.append(frame)

# Convert frames to a tensor
frames_tensor = torch.stack(frames)

# Load a pre-trained model
model = resnet50(pretrained=False, weights='imagenet')
model.eval()

# Process frames with the model
with torch.no_grad():
    features = model.conv1(frames_tensor)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)
    features = model.avgpool(features)
    features = torch.flatten(features, 1)

# Save the extracted images
output_path = '/home/hadmin/Allsafe/finetuning/videos-to-frames/images/'
for i in range(len(frames)):
    image_path = '{}frame_{}.jpg'.format(output_path, i)
    write_jpeg(frames[i], image_path)
