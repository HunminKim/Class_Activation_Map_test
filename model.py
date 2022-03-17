import torchvision.models as models
import torchvision.transforms as transforms
import torch

class ClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet101(pretrained=True)
        for name, param in self.model.named_parameters():
            if 'fc.weight' in name:
                self.fc_parameters = param

        self.featuremap_temp = {}
        self.model.layer4.register_forward_hook(self.get_activation('layer4'))
        self.image_transform =  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        

    def get_activation(self, name):
        def hook(model, input, output):
            self.featuremap_temp[name] = output.detach().cpu()
        return hook

    def forward(self, x):
        x = self.image_transform(x)
        result = self.model(x)
        self.featuremap = self.featuremap_temp['layer4']
        return result