# PRUEBA DEL TRANSFORMER CON BASE DE REGIONES FLANQUEANTES VENTANA 500, MAYÚSCULAS

import numpy as np
import pandas as pd
import torch
import gc
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback

base = pd.read_csv("/home/jbs1009/TFM/datosGene4PD/base_flanqueantes_500_OR_3109_con_rsid_extendida.csv", sep = ",", index_col = False)
etiqueta_binaria = {"Sano": 0, "Riesgo_PD": 1}
base["labels"] = base["labels"].map(etiqueta_binaria)

gss1 = GroupShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 2026)
id_tv, id_test = next(gss1.split(base, base["labels"], groups = base["rsID"]))

base_tv = base.iloc[id_tv].reset_index(drop = True)
base_test = base.iloc[id_test].reset_index(drop = True)

gss2 = GroupShuffleSplit(n_splits = 1, test_size = 0.25, random_state = 2026)
id_train, id_val = next(gss2.split(base_tv, base_tv["labels"], groups = base_tv["rsID"]))

base_train = base_tv.iloc[id_train].reset_index(drop = True)
base_val = base_tv.iloc[id_val].reset_index(drop = True)

hf_dataset = DatasetDict({
    "train": Dataset.from_pandas(base_train),
    "validation": Dataset.from_pandas(base_val),
    "test": Dataset.from_pandas(base_test)
})





tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

def tokeniza_secuencias(batch):
    return tokenizer([s.upper() for s in batch["Secuencia"]])

hf_base_tokenizada = hf_dataset.map(
    tokeniza_secuencias, 
    batched=True, 
    batch_size=10  
)
gc.collect() 


class_weights = compute_class_weight("balanced", classes = np.unique(base_train["labels"]), y = base_train["labels"])

weights_tensor = torch.tensor([1.0, 10.0], dtype = torch.float32)


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight = weights_tensor.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        logits = predictions[0]
    
    else:
        logits = predictions

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim = -1).numpy()[:, 1]
    preds = np.argmax(logits, axis = 1)

    return {
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "f1": f1_score(labels, preds)
    }

config = AutoConfig.from_pretrained(
    "zhihan1996/DNABERT-2-117M", 
    trust_remote_code=True, 
    num_labels=2
)

config.pad_token_id = tokenizer.pad_token_id
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

model = AutoModelForSequenceClassification.from_pretrained(
    "zhihan1996/DNABERT-2-117M", 
    config=config,
    trust_remote_code=True
)

for name, param in model.named_parameters():
    if "classifier" in name or "score" in name:
        param.requires_grad = True

    else:
        param.requires_grad = False

# parametros_entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Parámetros entrenables: {parametros_entrenables}")

data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

training_args = TrainingArguments(
    output_dir="./resultados_dnabert_flancos",  
    learning_rate=1e-5,                 
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      
    per_device_eval_batch_size=8,
    num_train_epochs=1000,
    warmup_ratio=0.1,                 
    weight_decay=0.01,
    lr_scheduler_type="cosine",


    evaluation_strategy="epoch",             
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=2,

    
    fp16=True,                   
    dataloader_num_workers=0,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    greater_is_better=True    
)

# trainer = WeightedTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=hf_base_tokenizada["train"],
#     eval_dataset=hf_base_tokenizada["validation"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
# )

# trainer = WeightedTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=hf_base_tokenizada["train"],
#     eval_dataset=hf_base_tokenizada["validation"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics
# )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_base_tokenizada["train"],
    eval_dataset=hf_base_tokenizada["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


trainer.train()



print("EVALUACIÓN EN CONJUNTO DE TEST")

predicciones = trainer.predict(hf_base_tokenizada["test"])

if isinstance(predicciones.predictions, tuple):
    logits_test = predicciones.predictions[0]

else:
    logits_test = predicciones.predictions

labels_test = predicciones.label_ids

probs_test = torch.nn.functional.softmax(torch.tensor(logits_test), dim = 1).numpy()[:, 1]
preds_test = np.argmax(logits_test, axis = 1)

print(f"ROC-AUC Test: {roc_auc_score(labels_test, probs_test):.4f}")
print(f"PR-AUC Test: {average_precision_score(labels_test, probs_test):.4f}\n")

print(classification_report(labels_test, preds_test, target_names = ["Sano", "Riesgo_PD"]))