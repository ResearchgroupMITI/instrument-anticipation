import torchvision.models as models
import torch.nn as nn
#from torchsummary import summary
# feel free to add to the list: https://pytorch.org/docs/stable/torchvision/models.html

### TWO HEAD MODELS ###

class ResNet50Model(nn.Module):
    def __init__(self, hparams):
        super(ResNet50Model, self).__init__()
        if hparams.pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = models.resnet50(weights=weights)
        # replace final layer with number of labels
        self.model.fc = Identity()
        self.fc_features = nn.Linear(2048, hparams.num_features)

    def forward(self, x):
        out_stem = self.model(x)
        features = self.fc_features(out_stem)
        return out_stem, features

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet50model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        resnet50model_specific_args.add_argument("--pretrained",
                                                 action="store_true",
                                                 help="pretrained on imagenet")
        return parser


#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
