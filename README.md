# weather-prediction

## Repository structure
```
- data/  
- notebooks/  
    |__ rain-prediction.ipynb  
- results/  
    |__plots/...  
- src/  
    |__base/  
        |__data_base.py  
        |__dataset.py  
    |__rainpred/  
        |__data.py  
        |__feature_generator.py  
        |__model_factory.py  
        |__prepocessor_factory.py  
    |__utils/  
        |__load_kaggle_data.py  
        |__visualisations.py  
    |__run_rain_prediction.py  
- .gitignore  
- README.md  
- requirements.txt  
```
## Rain Prediction Module
### Console Interface
To run the prediction for the [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/data) dataset, use the following script.
```
python src/run_rain_prediction.py [--analyze, -a] [--visualise, -v] [--result_path [argument]] [--download_data, -d]
```
The trained models are:
- DecisionTreeClassifier  
- RandomForestClassifier  
- MLPClassifier  

