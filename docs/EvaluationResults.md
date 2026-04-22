# Explore training_run results


```python
import pandas as pd
```


```python
df = pd.read_csv("./training_runs.csv", sep=",")
```

Comprobamos cuantos runs tenemos


```python
print("Numero de runs : ",str(df.shape[0]))
print("Por caso de uso - experimento")
df.value_counts("experiment_id")
```

    Numero de runs :  80
    Por caso de uso - experimento
    




    experiment_id
    3    36
    1    34
    2    10
    Name: count, dtype: int64



recogemos las columnas de metadatado y versionado


```python
tag = df.columns[df.columns.str.match('^tag')]
param_cols = df.columns[df.columns.str.match('^param')].tolist()
metadata_model_version =  param_cols
```

## USE CASE 1 - THERAPY MODELLING

Recogemos cual ha sido nuestro mejor modelo seleccionado en base a la métrica `f1_samples` ( debido al debalance de clases ) entre todas las evaluaciones por combinatorias cohorte


```python
df_multilabel = df[(df["experiment_id"]==1) & (df["param.model_selection_shap_skipped"]!="TRUE")]

metrics_therapy = df_multilabel.columns[(df_multilabel.columns.str.match('.*therapy.*')) | ( df_multilabel.columns.str.match('.*surgery.*'))].tolist()
metrics_therapy =  metrics_therapy + ['metric.test_f1_macro','metric.test_f1_samples','metric.test_subset_accuracy']

print("Comparacion con f1 ponderada")
mean_scores_f1_samples = df_multilabel.groupby(['param.selected_model'])['metric.test_f1_samples'].mean().idxmax
print(mean_scores_f1_samples)

print("\nComparacion con f1 global")
mean_scores_f1_macro = df_multilabel.groupby(['param.selected_model'])['metric.test_f1_macro'].mean().idxmax
print(mean_scores_f1_macro)

print("\nComparacion con acc")
mean_scores_acc = df_multilabel.groupby(['param.selected_model'])['metric.test_subset_accuracy'].mean().idxmax
print(mean_scores_acc)
```

    Comparacion con f1 ponderada
    <bound method Series.idxmax of param.selected_model
    random_forest    0.676429
    xgboost          0.643378
    Name: metric.test_f1_samples, dtype: float64>
    
    Comparacion con f1 global
    <bound method Series.idxmax of param.selected_model
    random_forest    0.673385
    xgboost          0.682073
    Name: metric.test_f1_macro, dtype: float64>
    
    Comparacion con acc
    <bound method Series.idxmax of param.selected_model
    random_forest    0.231432
    xgboost          0.193184
    Name: metric.test_subset_accuracy, dtype: float64>
    

RF trata mejor clases minoritarias o balance

XGB optimiza rendimiento global

Ya que el f1 global no es considerablemente mayor en XGBOOST y queremos que se ajuste lo mejor a las clases minoritarias el modelo, elegimos el Random Forest.

En cuanto a precisión global, es demasiado bajo para utilizarlo en la práctica

Vemos con más detalle la precisión por clase y cohorte en Random Forest


```python
eval_cdu_1 = (df_multilabel[metadata_model_version+metrics_therapy]
.drop(['param.inner_cv_splits','param.inner_cv_strategy','param.model_selection_shap_skipped'], axis=1)
.drop_duplicates()).dropna()
```


```python
eval_cdu_1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>param.held_out_cohort</th>
      <th>param.selected_model</th>
      <th>metric.breast_surgery_BREAST_CONSERVING_accuracy</th>
      <th>metric.breast_surgery_BREAST_CONSERVING_f1_binary</th>
      <th>metric.breast_surgery_MASTECTOMY_accuracy</th>
      <th>metric.breast_surgery_MASTECTOMY_f1_binary</th>
      <th>metric.chemotherapy_accuracy</th>
      <th>metric.chemotherapy_f1_binary</th>
      <th>metric.hormone_therapy_accuracy</th>
      <th>metric.hormone_therapy_f1_binary</th>
      <th>metric.radio_therapy_accuracy</th>
      <th>metric.radio_therapy_f1_binary</th>
      <th>metric.test_f1_macro</th>
      <th>metric.test_f1_samples</th>
      <th>metric.test_subset_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>5.0</td>
      <td>xgboost</td>
      <td>0.653465</td>
      <td>0.477612</td>
      <td>0.643564</td>
      <td>0.731343</td>
      <td>0.950495</td>
      <td>0.915254</td>
      <td>0.683168</td>
      <td>0.636364</td>
      <td>0.415842</td>
      <td>0.535433</td>
      <td>0.659201</td>
      <td>0.602876</td>
      <td>0.207921</td>
    </tr>
    <tr>
      <th>31</th>
      <td>4.0</td>
      <td>xgboost</td>
      <td>0.632768</td>
      <td>0.551724</td>
      <td>0.655367</td>
      <td>0.670270</td>
      <td>0.909605</td>
      <td>0.804878</td>
      <td>0.666667</td>
      <td>0.742358</td>
      <td>0.587571</td>
      <td>0.724528</td>
      <td>0.698752</td>
      <td>0.676110</td>
      <td>0.158192</td>
    </tr>
    <tr>
      <th>32</th>
      <td>3.0</td>
      <td>xgboost</td>
      <td>0.590909</td>
      <td>0.439024</td>
      <td>0.592885</td>
      <td>0.677116</td>
      <td>0.956522</td>
      <td>0.887755</td>
      <td>0.679842</td>
      <td>0.716783</td>
      <td>0.590909</td>
      <td>0.720648</td>
      <td>0.688265</td>
      <td>0.651148</td>
      <td>0.213439</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2.0</td>
      <td>random_forest</td>
      <td>0.615385</td>
      <td>0.540816</td>
      <td>0.628205</td>
      <td>0.639004</td>
      <td>0.987179</td>
      <td>0.969072</td>
      <td>0.722222</td>
      <td>0.765343</td>
      <td>0.717949</td>
      <td>0.833333</td>
      <td>0.749514</td>
      <td>0.710277</td>
      <td>0.307692</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1.0</td>
      <td>random_forest</td>
      <td>0.665025</td>
      <td>0.517730</td>
      <td>0.667488</td>
      <td>0.732673</td>
      <td>0.640394</td>
      <td>0.223404</td>
      <td>0.672414</td>
      <td>0.743738</td>
      <td>0.642857</td>
      <td>0.768740</td>
      <td>0.597257</td>
      <td>0.642582</td>
      <td>0.155172</td>
    </tr>
  </tbody>
</table>
</div>



## USE CASE 2 - SURVIVAL MODELLING

Recogemos cual ha sido nuestro mejor modelo seleccionado en base a la métrica `r2` ( debido al debalance de clases ) entre todas las evaluaciones por combinatorias cohorte


```python
df_regression = df[df["experiment_id"]==2]

metrics_prog =   ['metric.test_rmse','metric.test_mae','metric.test_r2']

print("Comparacion con R2")
mean_scores_f1_samples = df_regression.groupby(['param.selected_model'])['metric.test_r2'].mean().idxmax
print(mean_scores_f1_samples)

print("\nComparacion con MAE")
mean_scores_f1_macro = df_regression.groupby(['param.selected_model'])['metric.test_mae'].mean().idxmax
print(mean_scores_f1_macro)

print("\nComparacion con RMSE")
mean_scores_acc = df_regression.groupby(['param.selected_model'])['metric.test_rmse'].mean().idxmax
print(mean_scores_acc)
```

    Comparacion con R2
    <bound method Series.idxmax of param.selected_model
    random_forest    0.136558
    Name: metric.test_r2, dtype: float64>
    
    Comparacion con MAE
    <bound method Series.idxmax of param.selected_model
    random_forest    47.204016
    Name: metric.test_mae, dtype: float64>
    
    Comparacion con RMSE
    <bound method Series.idxmax of param.selected_model
    random_forest    59.360406
    Name: metric.test_rmse, dtype: float64>
    

Los resultados no indican capacidad predictiva limitada, puesto que el único modelo que ha superado el entrenamiento de regresion es  explica solo 13.7% de la varianza.

## USE CASE 3 - DIAGNOSIS MODELLING

Recogemos cual ha sido nuestro mejor modelo seleccionado en base a la métrica `f1_weighted`


```python
df_multiclass = df[(df["experiment_id"]==3)&(~df["tag.inner_strategy"].isna())]

metrics_diagnosis = df_multiclass.columns[(df_multiclass.columns.str.match('.*ctree.*'))].tolist()
metrics_diagnosis =  metrics_diagnosis + ['metric.test_subset_accuracy']

print("Comparacion con precision")
mean_scores_f1_samples = df_multiclass.groupby(['param.selected_model'])['metric.test_exact_match_accuracy'].mean().idxmax
print(mean_scores_f1_samples)

```

    Comparacion con precision
    <bound method Series.idxmax of param.selected_model
    xgboost    0.989347
    Name: metric.test_exact_match_accuracy, dtype: float64>
    


```python
eval_cdu_3 = (df_multiclass[metadata_model_version+metrics_diagnosis]
.drop(['param.inner_cv_splits','param.inner_cv_strategy','param.model_selection_shap_skipped'], axis=1)
.drop_duplicates()[0:5])
eval_cdu_3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>param.held_out_cohort</th>
      <th>param.selected_model</th>
      <th>metric.orctree_breast_accuracy</th>
      <th>metric.orctree_breast_f1_weighted</th>
      <th>metric.orctree_idc_accuracy</th>
      <th>metric.orctree_idc_f1_weighted</th>
      <th>metric.orctree_ilc_accuracy</th>
      <th>metric.orctree_ilc_f1_weighted</th>
      <th>metric.orctree_immc_accuracy</th>
      <th>metric.orctree_immc_f1_weighted</th>
      <th>metric.orctree_mbc_accuracy</th>
      <th>metric.orctree_mbc_f1_weighted</th>
      <th>metric.orctree_mdlc_accuracy</th>
      <th>metric.orctree_mdlc_f1_weighted</th>
      <th>metric.orctree_nan_accuracy</th>
      <th>metric.orctree_nan_f1_weighted</th>
      <th>metric.test_subset_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>xgboost</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>xgboost</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.995763</td>
      <td>0.995788</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.978814</td>
      <td>0.968334</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>xgboost</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.991826</td>
      <td>0.991865</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.990463</td>
      <td>0.985718</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>xgboost</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.993007</td>
      <td>0.993039</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.996503</td>
      <td>0.994758</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.996503</td>
      <td>0.998249</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>xgboost</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.993763</td>
      <td>0.993774</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.991684</td>
      <td>0.990307</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




en este caso la estrategia de uso de esta serie de modelos ha sido correcta, ya que se traduce a todas las combinaciones de cohortes con buenas métricas
