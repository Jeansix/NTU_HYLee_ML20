import numpy as np
import pandas as pd
from model import *
import torch


def inference(test_dataloader, extractor_dir, predictor_dir):
    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()
    feature_extractor.load_state_dict(torch.load(extractor_dir))
    label_predictor.load_state_dict(torch.load(predictor_dir))
    feature_extractor.eval()
    label_predictor.eval()

    result = []
    label_predictor.eval()
    feature_extractor.eval()
    for i, (test_data, _) in enumerate(test_dataloader):
        test_data = test_data.cuda()

        class_logits = label_predictor(feature_extractor(test_data))

        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)
    result = np.concatenate(result)

    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
    df.to_csv('DaNN_submission.csv', index=False)
