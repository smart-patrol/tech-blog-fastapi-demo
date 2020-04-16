import torchvision.transforms as transforms
from PIL import Image


def transform_image(image_path):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(image_path)
    return my_transforms(image).unsqueeze(0)


def get_prediction(model, image_path):
    tensor = transform_image(image_path=image_path)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

