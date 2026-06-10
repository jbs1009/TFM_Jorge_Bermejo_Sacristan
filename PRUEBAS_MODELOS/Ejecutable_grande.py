# EJECUTABLE GENERAL CON TODOS LOS MODELOS IMPLEMENTADOS PARA EJECUTAR EN SEGUNDO PLANO EN EL SERVIDOR DE MANERA EFICIENTE Y OBTENER RESULTADOS FINALES

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

tf.config.set_visible_devices([], 'GPU')

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, 
                             precision_score, recall_score, matthews_corrcoef, confusion_matrix)

from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback)


DATASETS = {
    "Ventana_50_Extendida": "/home/jbs1009/TFM/datosGene4PD/base_flanqueantes_50_OR_3109_con_rsid_extendida.csv",
    "Ventana_50_Reducida": "/home/jbs1009/TFM/datosGene4PD/base_flanqueantes_50_OR_2400_reducida_extendida.csv",
    "Ventana_500_Extendida": "/home/jbs1009/TFM/datosGene4PD/base_flanqueantes_500_OR_3109_con_rsid_extendida.csv", 
    "Ventana_500_Reducida": "/home/jbs1009/TFM/datosGene4PD/base_flanqueantes_500_OR_2400_reducida_extendida.csv"   
}


etiqueta_binaria = {"Sano": 0, "Riesgo_PD": 1}

def preprocesar_base(ruta):
    base = pd.read_csv(ruta)
    if "Etiqueta" in base.columns:
        base = base.rename(columns={"Etiqueta": "labels"})
    if base["labels"].dtype == object:
        base["labels"] = base["labels"].map(etiqueta_binaria)
    return base

def extract_metrics(y_true, y_pred_proba, y_pred_bin):
    
    precision = precision_score(y_true, y_pred_bin, zero_division=0)
    recall = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    mcc = matthews_corrcoef(y_true, y_pred_bin)
    
    return {
        "ROC_AUC": roc_auc_score(y_true, y_pred_proba),
        "PR_AUC": average_precision_score(y_true, y_pred_proba),
        "MCC": mcc,
        "F1_Riesgo": f1,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity
    }


def ejecutar_modelos_clasicos(base, dataset_name):
    print(f"\n--- Ejecutando Random Forest y MLP para {dataset_name} ---")
    resultados = []
    X_seq = base["Secuencia"]
    y = base["labels"].values
    grupos_rsid = base["rsID"].values

    encoding = {"a": [1, 0, 0, 0], "c": [0, 1, 0, 0], "g": [0, 0, 1, 0], "t": [0, 0, 0, 1]}
    def one_hot_encode_sequence(seq):
        return np.array([encoding.get(nuc.lower(), [0, 0, 0, 0]) for nuc in seq])
    
    X_oh = np.array([one_hot_encode_sequence(seq) for seq in X_seq])
    X_oh = X_oh.reshape(X_oh.shape[0], -1)

    def kmeriza(sequence, k=3):
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    kmers_series = X_seq.apply(lambda seq: ' '.join(kmeriza(seq.lower(), k=3)))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=2026)
    train_id, test_id = next(gss.split(X_seq, y, groups=grupos_rsid))

    y_train, y_test = y[train_id], y[test_id]
    X_train_oh, X_test_oh = X_oh[train_id], X_oh[test_id]

    vectorizer = CountVectorizer()
    X_train_kmers = vectorizer.fit_transform(kmers_series.iloc[train_id]).toarray()
    X_test_kmers = vectorizer.transform(kmers_series.iloc[test_id]).toarray()

    scaler = StandardScaler()
    X_train_kmers_scaled = scaler.fit_transform(X_train_kmers)
    X_test_kmers_scaled = scaler.transform(X_test_kmers)

    pesos_fijos = {0: 1.0, 1: 10.0}

    escenarios = [
        ("RandomForest", "One-Hot", RandomForestClassifier(n_estimators=200, class_weight=pesos_fijos, random_state=2026, n_jobs=-1), X_train_oh, X_test_oh),
        ("RandomForest", "K-mers", RandomForestClassifier(n_estimators=200, class_weight=pesos_fijos, random_state=2026, n_jobs=-1), X_train_kmers, X_test_kmers),
        ("MLP", "One-Hot", MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=2026), X_train_oh, X_test_oh),
        ("MLP", "K-mers", MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=2026), X_train_kmers_scaled, X_test_kmers_scaled)
    ]

    for model_name, feat_name, model, x_tr, x_ts in escenarios:
        model.fit(x_tr, y_train)
        y_pred_proba = model.predict_proba(x_ts)[:, 1]
        y_pred_bin = model.predict(x_ts)
        mets = extract_metrics(y_test, y_pred_proba, y_pred_bin)
        resultados.append({"Dataset": dataset_name, "Model": model_name, "Feature_Extraction": feat_name, **mets})
    
    return resultados

def ejecutar_lstm(base, dataset_name):
    print(f"\n--- Ejecutando LSTM Bidireccional para {dataset_name} ---")
    
    X_seq = base["Secuencia"]
    y = base["labels"].values
    grupos_rsid = base["rsID"].values

    dicc = {"a": 1, "c": 2, "g": 3, "t": 4}
    def codificador(secuencia):
        return np.array([dicc.get(nuc.lower(), 0) for nuc in secuencia])
    X_cod = np.array([codificador(secuencia) for secuencia in X_seq])

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
    id_train_val, id_test = next(gss1.split(X_cod, y, groups=grupos_rsid))

    X_tv, y_tv, grupos_tv = X_cod[id_train_val], y[id_train_val], grupos_rsid[id_train_val]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=2026)
    id_train, id_val = next(gss2.split(X_tv, y_tv, groups=grupos_tv))

    X_train, y_train = X_tv[id_train], y_tv[id_train]
    X_val, y_val = X_tv[id_val], y_tv[id_val]
    X_test, y_test = X_cod[id_test], y[id_test]

    dicc_pesos = {0: 1.0, 1: 10.0}

    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=5, output_dim=16, input_length=len(X_seq.iloc[0])),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32, dropout=0.3, recurrent_dropout=0.3)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=1, activation="sigmoid")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc_pr", curve="PR")])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, 
              class_weight=dicc_pesos, callbacks=[early_stop], verbose=0)

    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)
    mets = extract_metrics(y_test, y_pred_proba, y_pred_bin)
    
    return [{"Dataset": dataset_name, "Model": "Bi-LSTM", "Feature_Extraction": "Int_Encoding", **mets}]

def ejecutar_transformer(base, dataset_name):
    print(f"\n--- Ejecutando DNABERT-2 para {dataset_name} ---")
    resultados = []
    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
    id_tv, id_test = next(gss1.split(base, base["labels"], groups=base["rsID"]))
    base_tv = base.iloc[id_tv].reset_index(drop=True)
    base_test = base.iloc[id_test].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=2026)
    id_train, id_val = next(gss2.split(base_tv, base_tv["labels"], groups=base_tv["rsID"]))
    base_train = base_tv.iloc[id_train].reset_index(drop=True)
    base_val = base_tv.iloc[id_val].reset_index(drop=True)

    hf_dataset = DatasetDict({
        "train": Dataset.from_pandas(base_train),
        "validation": Dataset.from_pandas(base_val),
        "test": Dataset.from_pandas(base_test)
    })

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    def tokeniza_secuencias(batch):
        return tokenizer([s.upper() for s in batch["Secuencia"]])

    hf_base_tokenizada = hf_dataset.map(tokeniza_secuencias, batched=True, batch_size=10)

    weights_tensor = torch.tensor([1.0, 10.0], dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(model.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        logits = predictions[0] if isinstance(predictions, tuple) else predictions
        
        probs = torch.nn.functional.softmax(torch.tensor(logits.astype(float)), dim=-1).numpy()[:, 1]
        preds = np.argmax(logits, axis=1)
        
        return {
            "roc_auc": roc_auc_score(labels, probs),
            "pr_auc": average_precision_score(labels, probs),
            "f1": f1_score(labels, preds)
        }

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for trainer_type in ["NormalTrainer", "WeightedTrainer"]:
        print(f"\nEntrenando DNABERT-2 con {trainer_type}...")
        
        model = AutoModelForSequenceClassification.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, num_labels=2, ignore_mismatched_sizes=True)
        
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.hidden_dropout_prob = 0.2
        model.config.attention_probs_dropout_prob = 0.2

        training_args = TrainingArguments(
            output_dir=f"./resultados_dnabert_{dataset_name}_{trainer_type}", 
            learning_rate=1e-5, per_device_train_batch_size=4, gradient_accumulation_steps=4, 
            per_device_eval_batch_size=8, num_train_epochs=50, weight_decay=0.01, lr_scheduler_type="cosine",
            evaluation_strategy="epoch", save_strategy="epoch", logging_strategy="epoch", save_total_limit=1,
            fp16=True, dataloader_num_workers=0, load_best_model_at_end=True, metric_for_best_model="roc_auc", greater_is_better=True
        )

        TrainerClass = WeightedTrainer if trainer_type == "WeightedTrainer" else Trainer

        trainer = TrainerClass(
            model=model,
            args=training_args,
            train_dataset=hf_base_tokenizada["train"],
            eval_dataset=hf_base_tokenizada["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()

        predicciones = trainer.predict(hf_base_tokenizada["test"])
        logits_test = predicciones.predictions[0] if isinstance(predicciones.predictions, tuple) else predicciones.predictions
        
        probs_test = torch.nn.functional.softmax(torch.tensor(logits_test.astype(float)), dim=1).numpy()[:, 1]
        preds_test = np.argmax(logits_test, axis=1)

        mets = extract_metrics(predicciones.label_ids, probs_test, preds_test)
        resultados.append({"Dataset": dataset_name, "Model": "DNABERT-2-117M", "Feature_Extraction": trainer_type, **mets})

        del model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

    return resultados

if __name__ == "__main__":
    tabla_final = []

    for dataset_name, path in DATASETS.items():
        print(f"\n{'='*50}\nINICIANDO EVALUACIÓN PARA: {dataset_name}\n{'='*50}")
        
        # 1. Carga de datos
        try:
            base = preprocesar_base(path)
        except Exception as e:
            print(f"Error fatal cargando el dataset {dataset_name}: {str(e)}")
            continue
            
        # 2. Modelos Clásicos
        try:
            res_clasicos = ejecutar_modelos_clasicos(base, dataset_name)
            tabla_final.extend(res_clasicos)
        except Exception as e:
            print(f"--> ERROR en Modelos Clásicos ({dataset_name}): {str(e)}")
            
        # 3. BiLSTM
        try:
            res_lstm = ejecutar_lstm(base, dataset_name)
            tabla_final.extend(res_lstm)
        except Exception as e:
            print(f"--> ERROR en LSTM ({dataset_name}): {str(e)}")
            
        # 4. Transformer
        try:
            res_transf = ejecutar_transformer(base, dataset_name)
            tabla_final.extend(res_transf)
        except Exception as e:
            print(f"--> ERROR en Transformer ({dataset_name}): {str(e)}")
            
        pd.DataFrame(tabla_final).to_csv("resultados_progresivos.csv", index=False)

    df_resultados = pd.DataFrame(tabla_final)
    print("\n\n=== TABLA DE RESULTADOS FINALES ===")
    print(df_resultados.to_string(index=False))
    df_resultados.to_csv("TABLA_RESULTADOS_FINALES_TFM.csv", index=False)
    print("\nMétricas guardadas exitosamente en 'TABLA_RESULTADOS_FINALES_TFM.csv'.")
