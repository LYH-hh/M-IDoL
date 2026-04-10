This repository contains an official implementation of "[M-IDoL: Information Decomposition for Modality-Specific and Diverse Representation Learning in Medical Foundation Model]". 

## How to Perform Fine-tune
---
Download AFiRe's [pre-trained weight]().
Load the weights to the ViT-B model:
```
weights_dict = torch.load("./M-IDoL.pth", map_location=torch.device('cuda'))
for key in list(weights_dict.keys()):
    new_key = key.replace('backbone.', '')
    weights_dict[new_key] = weights_dict.pop(key)
model.load_state_dict(weights_dict, strict=False)
```
Our fine-tuning multi-label classification tasks are referred to the official code from [here](https://github.com/funnyzhou/REFERS) multi-class classification tasks from [here](https://github.com/openmedlab/RETFound_MAE) and segmentation tasks are referred to [here](https://github.com/SZUHvern/MaCo).
