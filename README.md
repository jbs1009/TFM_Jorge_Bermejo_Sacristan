# TFM_Jorge_Bermejo_Sacristan
Repositorio destinado a la realización del Trabajo de Fin de Máster de Jorge Bermejo Sacristán del Máster Universitario en Ingeniería Biomédica de la UBU

# Estudio de los modelos de lenguaje genómico para la clasificación de SNPs asociados al Parkinson: obtención, procesamiento de datos y evaluación de métodos 

En este trabajo de fin de máster se pretende estudiar el uso de grandes modelos del lenguaje (LLM) 
basados en secuencias biológicas combinados con bases de datos públicas relacionadas con la 
enfermedad de Parkinson, para la clasificación de SNPs asociados a la misma. El objetivo será analizar el potencial de los modelos preentrenados de 
arquitectura tipo BERT empleando técnicas de transfer learning para ayudar en el diagnóstico de la enfermedad de Parkinson. 

## Abstract

La Enfermedad de Parkinson es la segunda enfermedad neurodegenerativa más común del sistema nervioso central, así como
el desorden del movimiento más frecuente. En este Trabajo de Fin de Máster se ha estudiado el potencial de modelos de lenguaje genómicos para la predicción de predisposición a la Enfermedad de Parkinson a partir de polimorfismos de un único nucleótido (SNP). Tras todo un proceso de adquisición, preprocesamiento, preparación de datasets de entrenamiento y puesta a punto del transformer DNABERT-2-117M, mediante técnicas de transfer learning, y a pesar de la complejidad que representa la identificación de riesgo de esta enfermedad multigénica y multifactorial a partir de estas condiciones tan concretas, se ha logrado afinar el modelo para superar un AUC-ROC de 0.85, demostrando su capacidad de aprender las diferencias entre secuencias sanas y secuencias afectadas.

## Workflow seguido en el trabajo

<img width="1917" height="1077" alt="workflow_tfm" src="https://github.com/user-attachments/assets/765c3276-1495-48e0-aa66-574d9c50c9d6" />

## Tabla comparativa de resultados finales

<img width="1297" height="823" alt="image" src="https://github.com/user-attachments/assets/1f3d5925-71f0-4bae-8e42-8ac06c9c7543" />



