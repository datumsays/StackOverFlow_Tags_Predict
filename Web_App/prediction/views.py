from django.shortcuts import render
from django.http import HttpResponse
from .forms import InputForm
from .models import Posts
from django.shortcuts import redirect

# Create your views here

def content_to_predict(request):
    if request.method == "POST":
        form = InputForm(request.POST)
        if form.is_valid():
            return show_prediction_result(request)
    else:
        form = InputForm()
    return render(request, 'prediction/content_input.html', {'form': form})

def show_prediction_result(request):
    return HttpResponse('hello')