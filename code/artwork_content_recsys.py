# +
# Importing required packages
import os
import re
import nltk
import time
import string
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from nltk.corpus import wordnet, stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import pdist, jaccard, squareform

pd.set_option('mode.chained_assignment',None)


# -

class ArtworkContent:
    """
        Artwork Content Model training usiing Artwork features
        
    """

    def __init__(self, dataFrame):
        """
        initializing the dataframe
        
        """
        
        # renaming the raw column names to lower case names 
        dataFrame.rename(columns = {'ARTWORK_ID':'artwork_id', 'ARTIST_ID':'artist_id', 'ROOM_ID':'room_id', 
                                        'ARTWORK_PRICE':'artwork_price', 'GALLERY_ID':'gallery_id', 'ARTWORK_MEDIUM':'artwork_medium',
                                        'MATERIALS':'materials', 'ARTWORK_YEAR':'artwork_year', 'GALLERY_ESTABLISHED_YEAR':'gallery_established_year'
                           }, inplace = True)
        self.dataFrame = dataFrame
        
        
    def preprocessing(self, features):
        """
        Preprocessing the input dataframe for the training process
        
        ---------- Input ---------------
        features: The feature that needs to be considered for calculating the similarity
        
        ---------- output ---------------
        data: Preprocessed datafram
        nan_values: missing feature values
        
        """
        
        data = self.dataFrame[features]
        artwork_features = features
        artwork_features.remove('artwork_id')
        
        # finding the missing values in the features
        nan_values = {i: list(data[data[i].isna()]['artwork_id']) for i in artwork_features}
        
        # imputing the missing values
        data = self.impute_missing(data, nan_values, impute=True)
        
        # dropping null rows and duplicate rows
        data = data.dropna()
        data = data.drop_duplicates()
        print(f'Dataset shape: {data.shape}')

        # indexing Artwork id for the dataframe
        data.set_index('artwork_id', inplace=True)

        return data, nan_values

    def impute_missing(self, data, nan_values, impute = True):
        """
        Imputes missing values
        
        ---------- Input ---------------
        data: Preprocessed datafram
        nan_values: missing feature values
        impute: imputes value if True, remove imputed values if False
        
        ---------- output ---------------
        data: Imputed datafram
        
        """        
        
        if impute:
            
            # defining dummy values to impute
            impute_values ={
            'artist_id' : 0, 'artwork_medium' : 'dummy_medium', 'materials' : 'dummy_materials',
            'artwork_year' : int(data.artwork_year.mean()), 'artwork_price' : int(data.artwork_price.mean())
            }
            # imputing dummy values
            for item, value in nan_values.items():
                if len(value) > 0:
                    data.loc[data['artwork_id'].isin(value), item] = impute_values[item]
        else:
            data = data.reset_index()
            # removing the imputed values
            for item, value in nan_values.items():
                if len(value) > 0:
                    data.loc[data['artwork_id'].isin(value), item] = np.nan
            data.set_index('artwork_id', inplace=True)
        
        return data
        
    def clean_text(self, txt):
        """
        cleans the given text
        
        ---------- Input ---------------
        txt: text to be cleaned
        
        ---------- output ---------------
        txt: cleaned text
        
        """   
            
        stopwords = nltk.corpus.stopwords.words('english')
        txt = txt.lower()
        txt = "".join([c for c in txt if c not in string.punctuation])
        tokens = re.split('\W+', txt)
        txt = [word for word in tokens if word not in stopwords]
        return txt
        
    def get_wordnet_pos(self, word):
        """
        Find the parts of speech of a word
        
        ---------- Input ---------------
        word: the word 
        
        ---------- output ---------------
        tag_dict.get(tag, wordnet.NOUN): pos of that word
        
        """  

        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)    
        
    def lemmatization(self, token_txt):
        """
        lemmitizes the text
        
        ---------- Input ---------------
        token_txt: text to be lemmitized
        
        ---------- output ---------------
        txt: lemmitized text
        
        """  
        
        lemmatizer = nltk.WordNetLemmatizer()
        txt = [lemmatizer.lemmatize(word,  pos = self.get_wordnet_pos(word)) for word in token_txt ]
        txt = ' '.join(str(e) for e in txt)
        return txt
        
    def common_data(self, list1, list2):
        """
        Finds if there any common data in two list
        
        ---------- Input ---------------
        list1: first list
        list2: second list
        
        ---------- output ---------------
        result: returns true if theres any common data in the list
        
        """ 
        result = False
        for x in list1:
            for y in list2:
                if x == y:
                    result = True
        return result
        
    def combine_results(self, results):
        """
        Combines  list of results in to one single result
        
        ---------- Input ---------------
        results: list of results
        
        ---------- output ---------------
        final_result: combination of all the results
        
        """ 
        
        final_result = []
        for i in results:
            temp = []
            for j in results:
                i.sort()
                j.sort()
                if i !=j:
                    if self.common_data(i, j):
                        temp.append(i+j)
            if len(temp) <= 0:
                temp.append(i)
            temp[0].sort()
            final_result.append(list(set(temp[0])))

        return final_result
        
    def choose_text(self, test, indices):
        """
        Chooses the final text
        
        ---------- Input ---------------
        test: datafram
        indices: similarity indices list
        
        ---------- output ---------------
        final_value: final lemmitized material name (grouped)
        
        """ 
        
        temp_df = test.loc[indices, :]
        counter = 0
        final_value = ''
        for i, r in temp_df.iterrows():
            if len(r['lemmatized_materials']) > counter:
                counter = len(r['lemmatized_materials'])
                final_value = r['lemmatized_materials']

        return final_value
        
        
    def nlp_material(self, data):
        """
        convert the material text into cleaned, lemmitized and groped values
        
        ---------- Input ---------------
        data: dataframe
        
        ---------- output ---------------
        data: dataframe with grouped value in the material column
        
        """ 
        
        # Data Cleaning
        data['cleaned_text'] = data['materials'].apply(lambda x : self.clean_text(x))

        # Material lemmatization
        data['lemmatized_materials'] = data['cleaned_text'].apply(lambda x : self.lemmatization(x))

        # Dropped cleaned_text
        data = data.drop('cleaned_text', 1)

        # Vectorizing and Finding the cosine similarity between them
        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(data['lemmatized_materials'])
        cosine_similarities = cosine_similarity(count_matrix, count_matrix) 

        np.fill_diagonal(cosine_similarities, 0)
        sim_df = pd.DataFrame(cosine_similarities, index=data.index, columns= data.index)


        # filtering out the items with similarity score greater than 0.5
        results = []
        for i, r in sim_df.iterrows():    
            result = [[i,items[0]] for items in r.iteritems() if items[1] >= 0.70] 
            if len(result) > 0:
                results.append(result[0])

        # Combining the results
        final_result = self.combine_results(results)

        # Sorting the final result and removing the duplicates
        final_result.sort()
        sim_list = list(k for k,_ in itertools.groupby(final_result))

        # Change to common value
        for indices in sim_list:
            value = self.choose_text(data, indices)
            for index in indices:
                data.loc[data.index == index, "lemmatized_materials"] = value

        # Replace empty space with underscore
        data['lemmatized_materials'] = data.lemmatized_materials.replace(' ', '_', regex=True)

        return data
        
    def bin_labelling(self, bins):
        """
        creating bin label dynamically
        
        ---------- Input ---------------
        bins: bins
        
        ---------- output ---------------
        bin_labels: dynamically created bin labels
        
        """ 
        temp_edge = ''
        bin_labels = []
        for idx, edge in enumerate(bins):
            if idx != 0:
                bin_labels.append(str(f'{int(temp_edge)}-{int(edge)}'))
            temp_edge = int(edge)
                
        return bin_labels

    def bucket_year(self, df):
        """
        bucketting years by discrete intervals
        
        ---------- Input ---------------
        df: dataframe
        
        ---------- output ---------------
        df: dataframe with year bucket
        
        """ 
        
        min_value = df['artwork_year'].min()
        max_value = df['artwork_year'].max()
        
        years = list(df.artwork_year.unique())
        bin_width = int(len(years)/10) + 1 # bin width = Decade
        
        bins = np.linspace(min_value,max_value,bin_width)
        labels = self.bin_labelling(bins)
        
        df['artwork_period'] = pd.cut(df['artwork_year'], bins=bins, labels=labels, include_lowest=True)
        df = df.drop('materials', 1)
        df = df.drop('artwork_year', 1)

        return df    
        
    def bucket_price(self, data):
        """
        bucketting price by Quantile-based discretization function
        
        ---------- Input ---------------
        data: dataframe
        
        ---------- output ---------------
        data: dataframe with year bucket
        
        """ 
        min_value = data['artwork_price'].min()
        max_value = data['artwork_price'].max()
        price = list(self.dataFrame.artwork_price.unique())
        bin_width = int(len(price)/15) + 1
        
        data['artwork_price_range'] = pd.qcut(data['artwork_price'], q=bin_width, duplicates='drop')
        data = data.drop('artwork_price', 1)

        return data
    
    def artist_split(self, data):
        """
        split artist id column into two or more columns
        if an artwork has two or more artists
        
        ---------- Input ---------------
        data: dataframe
        
        ---------- output ---------------
        data: dataframe with splitted artist id columns
        
        """ 
        data['artist_id'] = data['artist_id'].apply(str)
        
        artist_df = data['artist_id'].str.split(',', expand=True)
        artist_df = artist_df.fillna(value=np.nan)
        artist_df = artist_df.add_prefix('artist')
        
        data = pd.concat([data, artist_df], axis=1)
        data = data.drop('artist_id', 1)
        return data
        
    def one_hot_encoding(self, data, catergorical_columns):
        """
        Encoding the categorical columns using one hot encoding method
        
        ---------- Input ---------------
        data: dataframe
        catergorical_columns: columns to be encoded
        
        ---------- output ---------------
        data: encoded dataframe
        
        """ 
        return pd.get_dummies(data, columns = catergorical_columns)
        
    def similarity_results(self, encoded_data, metric):

        """
        Calculate the Similarity between two 
        Artwork/ Room based on the given content 
        with the given distance metric.

        ---------- Input ---------------
        dataFrame: Input content Dataframe

        encoded_data: Dataframe, One hot encoded dataframe of 
            the original dataframe.

        metric : str or function,
            The distance metric to use. 
            Distance metrics: 'cosine', 'dice', 'jaccard'.

        ---------- Output ---------------
        results: Dictionary, A dictionary containing the Key 
        value pair of item and Similar items

        """

        sim =  1 - pdist(encoded_data, metric)

        similarity_matrix = pd.DataFrame(squareform(sim), index=encoded_data.index, columns= encoded_data.index)

        results = {}

        for idx, row in similarity_matrix.iterrows():
            item_list = row.drop(labels = [idx])
            sorted_list = item_list.sort_values(ascending = False)    
            results[idx] = [items for items in sorted_list.iteritems()]  

        return results
        
    def sort_results(self, similarity_results):
        """
        sorts the final results based on the given priority
        
        ---------- Input ---------------
        similarity_results: similarity results
        
        ---------- output ---------------
        final_result: final sorted results
        
        """ 

        final_result = {}
        key = 'artwork_id'
        sort_features = ['artwork_medium', 'artwork_price', 'artwork_year']

        for i in tqdm(similarity_results.keys()):
            df = pd.DataFrame(similarity_results[i])
            df.rename( columns={0:'artwork_id', 1: 'score'}, inplace=True )

            final=pd.merge(df, self.dataFrame, how='left', left_on=[key], right_on=[key])
            inp = [list(self.dataFrame[self.dataFrame[key] == i][feature])[0] for feature in sort_features]

            final['same_medium'] =  np.where(final['artwork_medium'] == inp[0], 1, 0)
            final['price_sort'] =  abs(final['artwork_price'] - inp[1])
            final['year_sort'] =  abs(final['artwork_year'] - inp[2])

            priority_order = ['score', "same_medium", "year_sort", "price_sort"]
            ascending = (False, False, True, True)

            temp_df = final.sort_values(priority_order, ascending = ascending)
            temp_df = temp_df.reset_index()

            sorted_result = [[row['artwork_id'], row['score'], idx] for idx, row in temp_df.iterrows()]
            final_result[i] = sorted_result
            
        return  final_result
    
    def fit(self):
        """
        Driver function for trining the Artwork content model
        
        """ 
        start_time = time.time()
        print('\n------ Artwork content model training started ------')
#         model = ArtworkContent(dataFrame)

        features = ['artwork_id', 'artist_id', 'artwork_medium', 'materials', 'artwork_year', 'artwork_price']
        data, nan_values = self.preprocessing(features)

        data= self.nlp_material(data)
        data = self.bucket_year(data)
        data = self.bucket_price(data)
        data = self.artist_split(data)

        data.rename(columns = {'artist0':'artist_id', 'lemmatized_materials':'materials',
                                  'artwork_period':'artwork_year', 'artwork_price_range':'artwork_price'}, inplace = True)

        data = self.impute_missing(data, nan_values, impute=False)

        catergorical_columns = list(data.columns)
        print(f'Features: {catergorical_columns}')

        encoded_data = self.one_hot_encoding(data, catergorical_columns)
        print(f'Encoded data shape: {encoded_data.shape}')

        self.similarity_results = self.similarity_results(encoded_data, 'cosine')

        self.result = self.sort_results(self.similarity_results)

        print("Time taken to train Artwork content model: %s seconds" % (time.time() - start_time))
