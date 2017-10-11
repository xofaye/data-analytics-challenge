from django.shortcuts import render
from .forms import PredictPriceForm


# Create your views here.
def home(request):
    if request.method == 'POST':
        form =PredictPriceForm(request.POST)
        if form.is_valid():
            query_params = {}
            query_params['title__icontains'] = form.cleaned_data['query']
            if form.cleaned_data['min_price']:
                query_params['price__gte'] = form.cleaned_data['min_price']
            if form.cleaned_data['max_price']:
                query_params['price__lte'] = form.cleaned_data['max_price']
            if form.cleaned_data['location']:
                query_params['location__city__icontains'] = form.cleaned_data['location']

            return render(request, 'index.html', {'form': form})

    args = {}
    args['form'] = PredictPriceForm()
    return render(request, 'index.html', args)
