from google_drive_downloader import GoogleDriveDownloader as gdd
import moviepy.editor
from ibm_watson import SpeechToTextV1
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

import json
import os
import re
import math
import pandas as pd
import numpy as np
from io import StringIO
from collections import Counter
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher

class Video():

    def __init__(self, url, job_post):
        self.url = url
        self.job_post = job_post

    def video_text(self, url):
        self.file_id = url[32:65]
        self.destination_path = 'resume/candidate.mp4'
        gdd.download_file_from_google_drive(file_id=self.file_id, dest_path=self.destination_path)
        self.video = moviepy.editor.VideoFileClip(self.destination_path)
        self.audio = self.video.audio
        self.audio_path = 'resume/candidate.mp3'
        self.audio.write_audiofile(self.audio_path)
        self.api = IAMAuthenticator('FB-BtLEbKl4axbacSnO4x6DoDXq_VgRSvhTKdVKIfFfb')
        self.speech_to_text = SpeechToTextV1(authenticator=self.api)
        self.speech_to_text.set_service_url('https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/af924047-1418-4683-955f-4376ca2b7707')
        with open(self.audio_path, 'rb') as self.audio_file:
            self.result = self.speech_to_text.recognize(audio=self.audio_file, content_type='audio/mp3').get_result()
        self.audio_file.close()
        self.text = str()
        for i in range(0, len(self.result['results'])):
            self.text += self.result['results'][i]['alternatives'][0]['transcript']
        self.video.close()
        self.audio.close()
        return self.text

    def keyword_count(self, job_post):
        self.data_scientist = {'Statistics' : ['statistical models', 'statistical modeling', 'probability', 'normal distribution', 'poission distribution', 'survival models', 'hypothesis testing', 'bayesian inference', 'factor analysis', 'forecasting', 'markov chain', 'monte carlo', 'statistics', 'descriptive', 'inferential', 'classification', 'resampling', 'subset selection', 'shrinkage', 'dimesion reduction', 'nonlinear model', 'tree based'], 'MachineLearning' : ['linear regression', 'logistic regression', 'K means', 'random forest', 'xgboost', 'svm', 'naïve bayes', 'pca', 'decision trees', 'svd', 'ensemble models', 'boltzman machine', 'machine learning', 'hierarchical', 'support vector machine', 'principal component analysis', 'pca', 'least squares', 'polynomial fitting', 'conditional random field', 'ensembling', 'clustering', 'association', 'reinforcement', 'supervised', 'unsupervised', 'knn', 'apriori', 'bagging and random forests'], 'DeepLearning':['neural network', 'keras', 'theano', 'face detection', 'neural networks', 'convolutional neural network', 'recurrent neural network', 'object detection', 'yolo', 'gpu', 'cuda', 'tensorflow', 'lstm', 'gan', 'opencv', 'cnn', 'rnn', 'deep learning', 'feed forward neural network', 'adaboost', 'deep neural network', 'dnn', 'cnn', 'rnn', 'ffnn', 'transfer', 'gradient descent', 'deep convolution activation function', 'decaf', 'max pooling', 'batch normalization', 'skip gram', 'pytorch'], 'R' : ['ggplot', 'shiny', 'cran', 'dplyr', 'tidyr', 'lubridate', 'knitr', 'r', 'aesthetic', 'tibbles', 'relational ', 'model', 'pipe', 'vectors', 'iteration'], 'Python' : ['python', 'flask', 'django', 'pandas', 'numpy', 'scikitlearn', 'matplotlib', 'scipy', 'bokeh', 'statsmodel', 'seaborn', 'python', 'plotly', 'pydot', 'nltk', 'genism', 'scrapy', 'kivy', 'pyqt', 'opencv'], 'NaturalLanguageProcessing':['nlp', 'natural language processing', 'topic modeling', 'lda', 'named entity recognition', 'pos tagging', 'word2vec', 'word embedding', 'lsi', 'spacy', 'genism', 'nltk', 'nmf', 'doc2vec', 'cbow', 'bag of words', 'skip gram', 'bert', 'sentiment analysis', 'chat bot', 'chatbot'], 'DataEngineering' : ['aws', 'ec2', 'amazon redshift', 's3', 'docker', 'kubernetes', 'scala', 'teradata', 'goofle big query', 'aws lambda', 'aws emr', 'hive', 'hadoop', 'sql', 'data pipeline', 'data warehouse', 'etl', 'spark', 'redshift', 'mongodb', 'nosql', 'microsoft cloud azure', 'google cloud', 'postgresql', 'apache spark', 'mapreduce', 'kafka']}
        self.web_developer = {'WebTechnologies&FrameWork' : ['angular', 'html', 'css', 'kendo ui', 'php', 'mootools', 'dojo toolkit', 'xml', 'json', 'asp', 'sql', 'cakephp', 'codelgnitor', 'laravel', 'zend', 'yii', 'symfony', 'ruby', 'ruby on rails'], 'Scripts/UI' : ['javascript', 'oojs', 'jquery', 'ajax', 'bootstrap', 'angular js', 'node js', 'backbone js', 'express js', 'knoeckout js', 'react js', 'aws', 'firebas', 'magneto', 'wordpress', 'joomla', 'drupal', 'meteor js'], 'Database': ['mysql', 'hibernate', 'mongodb', 'redis', 'postgresql', 'oracle', 'sql server', 'nosql', 'datatooth', 'ibm db2', 'birt', 'dmy reports', 'vividcortex', 'airtable', 'clustercontrol', 'vertabelo', 'sap hana', 'mariadb', 'elasticsearch', 'cassandra', 'couchbase'], 'WebDebugger' : ['mozilla firebug', 'debugger', 'chrome developer', 'yslow', 'fiddler', 'httpwatch', 'colorzilla', 'fireshot', 'web inspector', 'fireftp', 'firebug', 'safari', 'internet explorer', 'debugbar', 'live http header', 'web accessibilty', 'venkman javascript', 'open dragonfly'], 'WebServer': ['apache tomcat', 'iis', 'nginx', 'litespeed', 'apache', 'node', 'lighttpd', 'apche http', 'caddy', 'oracle iplanet', 'cherokee', 'torando', 'gunicorn', 'monkey http server', 'xitami', 'mongrel', 'hiawatha', 'aol', 'navi', 'abyss', 'resin', 'puma', 'roxen', 'jexus'], 'Versioning':['git', 'bitbucket', 'jira', 'cvs', 'svn', 'mercurial', 'monotone', 'bazaar', 'tfs', 'vsts', 'perforce helix core', 'ibm rational clearcase', 'revision control system', 'visual sourcesafe', 'pvcs', 'darcs', 'accurev scm', 'vault', 'gnu arch', 'plastic scm', 'code coop'], 'Deployment':['docker', 'maven', 'cicd', 'jenkins']}
        self.android_developer = {'ProgrammingLanguage' : ['android ', 'android ndk', 'java', 'dagger', 'rx java', 'jni', 'c', 'j2ee', 'struts', 'javabeans', 'jsf', 'web services', 'spring', 'hibernate', 'jms', 'jdbc', 'javascript', 'soap', 'j unit', 'c++', 'swift', 'php', 'flutter', 'react native', 'ionic', 'xamarin'], 'MarkupLanguage': ['html', 'xml', 'css', 'xhtml', 'kotlin', 'c#', 'c sharp', 'corona', 'lua'], 'Servers': ['web logic ', 'apache tomcat', 'jboss', 'amazon web service mobile', 'firebase', 'parse'], 'BuildTools':['gradle', 'maven'], 'DebuggingTools': ['log cat', 'ddms', 'j unit', 'adb', 'android debugb bridge'], 'RDBMS': ['sqlite', 'db4o', 'oracle', 'mysql', 'ms sql server', 'realm db', 'ormlite', 'berkeley db', 'couchbase lite'], 'IDE':['android studio', 'eclipse', 'netbeans', 'intellij idea', 'avd manager', 'eclipse', 'fabric', 'flowup', 'gamemaker', 'leakcanary', 'nimbledroid', 'rad studio', 'stetho', 'source tree', 'unity 3d', 'unreal engine', 'visual studio', 'vysor'], 'OperatingSystems':['windows', 'unix', 'linux', 'macintosh']}
        self.job_dict = {'Data Scientist':self.data_scientist, 'Web Developer':self.web_developer, 'Android Developer':self.android_developer}
        for key in self.job_dict.keys():
            if key == job_post:
                self.post = self.job_dict[key]
                self.df = pd.DataFrame({keys:pd.Series(value) for keys, value in self.post.items()})
                self.total_keywords = 0
                for post_key in self.post.keys():
                    self.total_keywords += len(self.post[post_key])
                #print(self.total_keywords)
                return self.df, self.total_keywords
    
    def scoring_column(self, keyword_match, col_name, col_count):
        self.col_str = [] 
        for col in col_name:
            self.list_ =  re.findall('[A-Z][^A-Z]*', col)
            self.word = " ".join(self.list_)
            self.word = self.word.lower()
            self.col_str.append(self.word)
        self.col_name_count = {}
        for str_, count in zip(self.col_str, col_count):
            self.col_name_count[str_] = count
        #print(self.col_name_count)
        self.score = 0
        self.matched_keyword = []
        for keyword in keyword_match:
            for col in self.col_name_count.keys():
                if keyword == col:
                    self.value = self.col_name_count[col]
                    self.score += math.floor(self.value/2)
                    self.matched_keyword.append(keyword)      
        return self.score, self.matched_keyword

    def nlu(self, text, keyword):
        self.authenticator = IAMAuthenticator('LSSpBBaLUWaNu2X35ptYDGDmfoVTV5_seU2MXsiJs6yF')
        self.natural_language_understanding = NaturalLanguageUnderstandingV1(version='2019-07-12', authenticator=self.authenticator)
        self.natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/ad96f725-ca09-4e44-a5b9-55a64883b953')
    
        self.response = self.natural_language_understanding.analyze(text = text,
            features=Features(sentiment=SentimentOptions(targets=keyword))).get_result()

        self.dict_ = eval(json.dumps(self.response, indent=2))
    #print(dict_)
        self.negative_keyword = []
        for lst in self.dict_['sentiment']['targets']: 
            if lst['score'] <= 0.75:
                self.negative_keyword.append(lst['text'])
        return self.negative_keyword
    

    def subject_drop(self, df, keywords):
        print('exe')
        self.drop_subject = []
        for keyword in keywords:
            self.index = df[df['Keyword']==keyword].index.values
            self.subject = df['Subject'][self.index].tolist()
            self.drop_subject.append(self.subject[0])
    #print(drop_subject)
        for subject in self.drop_subject:
            self.df = df[df['Subject']!=subject]
        return self.df


    def create_profile(self, text, job_post):
        self.text = text.lower()
        self.keyword, self.total_keywords = self.keyword_count(job_post)
        self.columns_keyword = {}
        for column in self.keyword.columns:
            self.columns_keyword[column] = [nlp(text) for text in self.keyword[column].dropna(axis=0)]
        #print(self.columns_keyword)
    
        self.matcher = PhraseMatcher(nlp.vocab)
        for column in self.keyword.columns:
            self.matcher.add(column, None, *self.columns_keyword[column])
        self.doc = nlp(self.text)
        
        self.d = []  
        self.matches = self.matcher(self.doc)
        for match_id, start, end in self.matches:
            self.rule_id = nlp.vocab.strings[match_id]  
            self.span = self.doc[start : end] 
            self.d.append((self.rule_id, self.span.text))      
        self.keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(self.d).items())
        #print(keywords, type(keywords))
        
        self.df = pd.read_csv(StringIO(self.keywords),names = ['Keywords_List'])
        self.df1 = pd.DataFrame(self.df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
        self.df2 = pd.DataFrame(self.df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
        self.df3 = pd.concat([self.df1['Subject'],self.df2['Keyword'], self.df2['Count']], axis =1) 
        self.df3['Count'] = self.df3['Count'].apply(lambda x: x.rstrip(")"))
        
        self.dataf = pd.concat([self.df3['Subject'], self.df3['Keyword'], self.df3['Count']], axis = 1)
        if self.dataf.empty:
            return 0
        #print(self.dataf)
        self.dataf['Keyword'] = [i.strip() for i in self.dataf['Keyword']]
        self.negative_keyword = self.nlu(self.text, list(self.dataf['Keyword']))
        #print(self.negative_keyword)
        if len(self.negative_keyword) != 0:
            self.dataf = self.subject_drop(self.dataf, self.negative_keyword)
            if self.dataf.empty:
                return 0
        self.keyword_match = list(self.dataf['Keyword'])
        self.col_name = list(self.keyword.columns)
        self.col_count = list(self.keyword.count())
        self.column_score, self.matched_keyword = self.scoring_column(self.keyword_match, self.col_name, self.col_count)
        #print(self.column_score, self.matched_keyword)
        if len(self.matched_keyword) != 0:
            self.dataf = self.subject_drop(self.dataf, self.matched_keyword)
        #print(self.dataf)
        self.score = self.dataf['Keyword'].count()
        #print(self.score, self.column_score)
        self.profile_score = self.column_score + self.score
        #print(self.profile_score)
        self.final_score = self.profile_score / self.total_keywords
        return self.final_score

    def fit(self):
        self.text = self.video_text(self.url)
        try:
            self.score = self.create_profile(self.text, self.job_post)
        except:
            self.score = 0
        os.remove('resume/candidate.mp4')
        os.remove('resume/candidate.mp3')    
        return self.score

