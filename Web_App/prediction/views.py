from django.shortcuts import render
from django.http import HttpResponse
from .forms import InputForm
from .models import Posts
from django.shortcuts import redirect
from sklearn.externals import joblib
from preprocess import *
import numpy as np
import pandas as pd

# Create your views here

def content_to_predict(request):
    form = InputForm(request.POST)
    if form.is_valid():
        content = form.cleaned_data['content']
        return show_prediction_result(request, content)

    return render(request, 'prediction/content_input.html', {'form': form})

def show_prediction_result(request, data):

    # Clean out data
    text = stripTagsAndUris(data)
    text = removePunctuation(text)
    text = removeStopwords(text)
    text = stemmer(text)
    text = numeric_replacer(text)

    tfidf_selection = np.load('features.npy')#.item()

    print(tfidf_selection)
    print(text)
    testBow = GetBowDummies_Array2(pd.Series(text), tfidf_selection).index_feats_dict()

    print(testBow)

    mnb = joblib.load('mnb.pkl')
    pred = mnb.predict(testBow)
    result = {'topic': pred}
    return render(request, 'prediction/content_input.html', {'result': result})