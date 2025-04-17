import os
import pandas as pd
import random
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class IMDBDataset(Dataset):
    def __init__(self, base_path, size = 2000, train=True, transform=None):
        # pandas DataFrame
        self.df = self._load_imdb_data(base_path, size, train)
        self.features = list(self.df['review'])
        self.labels = LabelEncoder().fit_transform(self.df['sentiment'])
        self.transform = transform

    def __len__(self):
        return len(self.features) # number of samples

    def __getitem__(self, idx):
        feature = self.features[idx] #iloc
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        return feature, label
    
    @staticmethod
    def _load_imdb_data(base_path, sample_size, train):
        def score_to_sentiment(score):
            """Convert score to sentiment based on the given convention."""
            if score <= 4:
                return 'negative'
            elif score >= 7:
                return 'positive'
            else:
                return 'neutral'
            
        """Load IMDb data into a pandas DataFrame with balanced sampling."""
        data = []
        categories = ['pos', 'neg']
        sample_per_category = sample_size // 2  # Ensure equal sampling from each category

        for category in categories:
            if train == False: 
                # base_path/test/{neg or pos}/data_in_txt_format
                category_path = os.path.join(base_path, 'test', category)
            else:
                category_path = os.path.join(base_path, 'train', category)
            
            # read all file names in a list
            all_data_files = os.listdir(category_path) # ['1821_4.txt',  '9487_1.txt' ...]

            # get full path to each file in a list
            file_paths = [os.path.join(category_path, file_name) for file_name in all_data_files]

            # Randomly sample file paths from the current category
            sampled_files = random.sample(file_paths, sample_per_category)

            # Read each file and extract information
            for file_path in sampled_files:
                file_name = os.path.basename(file_path)
                id, score = file_name.split('_')
                score = int(score.replace('.txt', ''))
                sentiment = score_to_sentiment(score)

                with open(file_path, 'r', encoding='utf-8') as file:
                    review = file.read()

                data.append({'id': id, 'review': review, 'score': score, 'sentiment': sentiment})

        # Create DataFrame
        df = pd.DataFrame(data)
        return df
