import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import CLIPTextModel, CLIPTokenizer

class FiLMLayer(nn.Module):
    def __init__(self, channels, context_embedding_dim):
        super().__init__()
        self.generator = nn.Linear(context_embedding_dim, channels * 2)

    def forward(self, x, context):
        gamma_beta = self.generator(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        x = self.upsample(x)
        x = torch.cat([x, skip_features], dim=1)
        return self.conv_block(x)

class ModulatedUNet(nn.Module):
    def __init__(self, text_model_name="openai/clip-vit-base-patch32", freeze_encoders=True):
        super().__init__()
        self.text_model = CLIPTextModel.from_pretrained(text_model_name)
        self.text_embedding_dim = self.text_model.config.hidden_size
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder_pool1 = resnet.maxpool
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4
        
        if freeze_encoders:
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in resnet.parameters():
                param.requires_grad = False
            print("Encoders are frozen. Unfreezing BatchNorm layers in the image encoder.")
            for m in resnet.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
                    for param in m.parameters():
                        param.requires_grad = True

        self.film_layer = FiLMLayer(channels=512, context_embedding_dim=self.text_embedding_dim)
        self.decoder_layer4 = DecoderBlock(512, 256, 256)
        self.decoder_layer3 = DecoderBlock(256, 128, 128)
        self.decoder_layer2 = DecoderBlock(128, 64, 64)
        self.decoder_layer1 = DecoderBlock(64, 64, 64)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, image, text_tokens):
        original_size = image.shape[2:]
        text_outputs = self.text_model(**text_tokens)
        last_hidden_state = text_outputs.last_hidden_state
        attention_mask = text_tokens['attention_mask'].unsqueeze(-1).expand(last_hidden_state.shape)
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
        sum_mask = torch.sum(attention_mask, dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        text_embedding = sum_embeddings / sum_mask
        x = self.encoder_conv1(image)
        skip1 = x
        x = self.encoder_pool1(x)
        x = self.encoder_layer1(x)
        skip2 = x
        x = self.encoder_layer2(x)
        skip3 = x
        x = self.encoder_layer3(x)
        skip4 = x
        bottleneck = self.encoder_layer4(x)
        modulated_bottleneck = self.film_layer(bottleneck, text_embedding)
        x = self.decoder_layer4(modulated_bottleneck, skip4)
        x = self.decoder_layer3(x, skip3)
        x = self.decoder_layer2(x, skip2)
        x = self.decoder_layer1(x, skip1)
        x = self.final_upsample(x)
        logits = self.final_conv(x)
        logits_resized = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
        return logits_resized