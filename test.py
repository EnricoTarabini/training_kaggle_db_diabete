# Databricks notebook source
print('this is my first ML project on Databricks')

# COMMAND ----------

import os
diabetes = spark.read.format("csv").option("header","True").option("inferSchema","True").load(f"file:{os.getcwd()}/diabetes.csv")

# COMMAND ----------

diabetes.display()

# COMMAND ----------

display(diabetes.select('*').describe())

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'], outputCol='features')

output = assembler.transform(diabetes)

# COMMAND ----------

output.display()

# COMMAND ----------

diabetes_output = output.select('features','Outcome')
display(diabetes_output)

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures', withStd=True,withMean=True)

# COMMAND ----------

scaler_model = scaler.fit(diabetes_output)

# COMMAND ----------

scaled_data = scaler_model.transform(diabetes_output)
scaled_data.display()

# COMMAND ----------

from pyspark.ml.feature import Bucketizer

subset_output = output.select('Age')
splits = [0,10,20,30,40,50,60,70, float("inf")]

bucketizer = Bucketizer(splits=splits,inputCol='Age', outputCol='bucketedAge')
bucketed_data = bucketizer.transform(subset_output)
bucketed_data.display()

# COMMAND ----------

display(output)

# COMMAND ----------

features_outcome = output.select('features','Outcome')
features_outcome.display()

# COMMAND ----------

from pyspark.ml.feature import VectorSlicer

slicer = VectorSlicer(inputCol='features',outputCol='selectedFeatures', indices=[1,2,3,4,5,7])
features_subset = slicer.transform(features_outcome)
features_subset.display()

# COMMAND ----------

from pyspark.ml.feature import UnivariateFeatureSelector

selector = UnivariateFeatureSelector(selectionMode='numTopFeatures', featuresCol='features',outputCol='selectedFeatures',labelCol='Outcome')

selector.setFeatureType('continuous').setLabelType('categorical').setSelectionThreshold(1) #just the most relevant feature

# COMMAND ----------

selected_features = selector.fit(features_outcome).transform(features_outcome)

selected_features.display()

# COMMAND ----------

from pyspark.ml.feature import UnivariateFeatureSelector

selector = UnivariateFeatureSelector(selectionMode='percentile', featuresCol='features',outputCol='selectedFeatures',labelCol='Outcome')

selector.setFeatureType('continuous').setLabelType('categorical').setSelectionThreshold(0.5) #make PERCENTIL explicit

# COMMAND ----------

selected_features = selector.fit(features_outcome).transform(features_outcome)

selected_features.display()

# COMMAND ----------


