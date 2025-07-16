import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# --- Configurações ---
# Nome do arquivo CSV que você baixou do Kaggle
CSV_FILE_NAME = 'brazilian_headlines_sentiments.csv' # Corrigido para o nome que você usou
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 128 # Tamanho máximo da sequência (manchetes são curtas, 128 é um bom valor)
BATCH_SIZE = 16
NUM_EPOCHS = 3 # Você pode ajustar este valor

# --- 1. Carregar o Dataset ---
try:
    df = pd.read_csv(CSV_FILE_NAME)
    print(f"Dataset '{CSV_FILE_NAME}' carregado com sucesso.")
    print("Primeiras 5 linhas do DataFrame:")
    print(df.head())
    print("\nInformações sobre as colunas:")
    df.info()
except FileNotFoundError:
    print(f"Erro: O arquivo '{CSV_FILE_NAME}' não foi encontrado. Certifique-se de que ele está no mesmo diretório do seu script.")
    exit()

# --- 2. Limpeza e Seleção de Dados ---
# ATENÇÃO: As colunas corretas são 'headlinePortuguese' e 'sentimentScorePortuguese'
df_selected = df[['headlinePortuguese', 'sentimentScorePortuguese']].copy() # <-- CORREÇÃO AQUI

# Remover linhas com valores nulos, se houver
df_selected.dropna(inplace=True)

# Mapear os rótulos de sentimento para valores numéricos (0, 1, 2)
# O dataset tem 'sentimentScorePortuguese' que é um float.
# Vamos mapear scores:
#   Score < 0: Negativo (0)
#   Score == 0: Neutro (1)
#   Score > 0: Positivo (2)

def map_sentiment_score_to_label(score):
    if score < -0.1: # Se o score for menor que -0.1 (mais negativo)
        return 0 # Negativo
    elif score > 0.1: # Se o score for maior que 0.1 (mais positivo)
        return 2 # Positivo
    else: # Se o score estiver entre -0.1 e 0.1 (incluindo 0)
        return 1 # Neutro

df_selected['sentiment_id'] = df_selected['sentimentScorePortuguese'].apply(map_sentiment_score_to_label)

# Verifique se o mapeamento funcionou e a distribuição das classes
print("\nDistribuição dos sentimentos após mapeamento:")
print(df_selected['sentiment_id'].value_counts())
print(df_selected['sentimentScorePortuguese'].value_counts())

# O número de classes para o BERT
NUM_LABELS = df_selected['sentiment_id'].nunique()
print(f"\nNúmero de classes detectadas: {NUM_LABELS}")


# --- 3. Dividir os Dados ---
texts = df_selected['headlinePortuguese'].tolist() # <-- CORREÇÃO AQUI
labels = df_selected['sentiment_id'].tolist()

# Divisão em treino, validação e teste
# Usaremos 80% para treino e 20% para validação/teste
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
# Divide o conjunto temporário em validação e teste
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Número de exemplos de treino: {len(train_texts)}")
print(f"Número de exemplos de validação: {len(val_texts)}")
print(f"Número de exemplos de teste: {len(test_texts)}")

# --- 4. Carregar o Tokenizador BERT ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- 5. Tokenização dos Dados ---
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)
val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)
test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)

# --- 6. Criar Dataset no Formato PyTorch ---
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# --- 7. Carregar o Modelo BERT para Classificação ---
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Mover o modelo para a GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\nUsando dispositivo: {device}")

# --- 8. Função para Calcular Métricas ---
def compute_metrics(p):
    labels = p.label_ids
    preds = np.argmax(p.predictions, axis=1) # Pega o índice da classe com maior probabilidade
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 9. Configurar e Treinar o Trainer ---
training_args = TrainingArguments(
    output_dir='./results_headlines',
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_headlines',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none" # Desabilita o relatório para plataformas externas como Weights & Biases
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\n--- Iniciando o treinamento (Fine-tuning) do BERT ---")
trainer.train()
print("Treinamento concluído.")

# --- 10. Avaliar o Modelo no Conjunto de Teste ---
print("\n--- Avaliando o modelo no conjunto de TESTE ---")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Resultados da Avaliação no Teste: {test_results}")

# --- 11. Salvar o Modelo Fine-Tuned ---
model_save_path = "./modelo_bert_headlines_sentiment"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\nModelo fine-tuned salvo em '{model_save_path}'")

# --- 12. Exemplo de Inferência (usando o modelo treinado) ---
from transformers import pipeline

# Mapeamento reverso para exibir os nomes das classes
id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'} # <-- CORREÇÃO AQUI, pois sentiment_mapping mudou

classifier = pipeline(
    'text-classification',
    model=model_save_path, # Carrega o modelo salvo
    tokenizer=model_save_path, # Carrega o tokenizador salvo
    device=0 if torch.cuda.is_available() else -1
)

print("\n--- Exemplo de Classificação de Novas Manchetes ---")
new_headlines = [
    
    
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."



]

predictions = classifier(new_headlines)

for i, pred in enumerate(predictions):
    # O pipeline pode retornar 'LABEL_0', 'LABEL_1', 'LABEL_2'
    predicted_label_id = int(pred['label'].split('LABEL_')[-1])
    predicted_sentiment = id_to_label[predicted_label_id]
    score = pred['score']
    print(f"Manchete: \"{new_headlines[i]}\"")
    print(f"  Sentimento Previsto: {predicted_sentiment.capitalize()} (Confiança: {score:.4f})")
    print("-" * 50)