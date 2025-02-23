import os
import pandas as pd
import torch
import pyiqa
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

nr_metrics = ['qualiclip+']  # competing NR-IQA models


def nr_metric_test(metric_name):
    metric = pyiqa.create_metric(metric_name).to(device)
    print('lower_better:', metric.lower_better)

    df = pd.read_csv(score_file)

    for idx, images in tqdm(df.iterrows(), total=df.shape[0], desc=f"Computing 2AFC scores of {metric_name}"):
        group = images['group'].upper()
        pair = images['sr_pair_names']
        img1 = images['name1_column']
        img2 = images['name2_column']
        path1 = os.path.join('/home/user/study/SR', 'MAD', f'{group}_images', 'final', pair, img1)
        path2 = os.path.join('/home/user/study/SR', 'MAD', f'{group}_images', 'final', pair, img2)

        try:
            score1 = metric(path1).item()
            score2 = metric(path2).item()
        except AssertionError:
            img1 = Image.open(path1)
            img2 = Image.open(path2)
            w, h = img1.size
            if h < 224 or w < 224:
                scale_factor = 224 / min(h, w)
                new_h = int(h * scale_factor)
                new_w = int(w * scale_factor)
                img1 = img1.resize((new_w, new_h), Image.BICUBIC)
                img2 = img2.resize((new_w, new_h), Image.BICUBIC)
                score1 = metric(img1).item()
                score2 = metric(img2).item()

        if metric.lower_better:
            df.at[idx, metric_name] = images['score1_column'] if score1 < score2 else images['score2_column']
        else:
            df.at[idx, metric_name] = images['score2_column'] if score1 < score2 else images['score1_column']

    df.to_csv(score_file, index=False)


def main():
    for metric in tqdm(nr_metrics, desc="Testing metrics on MAD images"):
        nr_metric_test(metric)

    df = pd.read_csv(score_file)
    metrics = df.columns[6:]

    # Compute the upper and lower limits of model performance and human scores
    for idx, images in df.iterrows():
        df.at[idx, 'upper_limit'] = images[metrics].max()
        df.at[idx, 'lower_limit'] = images[metrics].min()
        df.at[idx, 'human'] = max(df.at[idx, 'score1_column'], df.at[idx, 'score2_column'])

    scores = df.columns[6:]
    means = df[scores].mean()
    # upper = means['upper_limit']
    # lower = means['lower_limit']
    # means = (means - lower) / (upper - lower)
    means = pd.DataFrame(means).T
    means.index = ['average']
    df = pd.concat([df, means])

    df.to_csv(score_file, index=True)


if __name__ == '__main__':
    score_file = './sr_subjective_results.csv'
    main()
