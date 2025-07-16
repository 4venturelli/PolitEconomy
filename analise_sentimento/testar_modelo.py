import torch
import torch.nn.functional as F # Importado para a função softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configurações ---
# Caminho para o modelo treinado que foi salvo pelo script 'treinar_bert.py'
MODEL_SAVE_PATH = "./modelo_bert_headlines_sentiment" # Certifique-se que esta pasta existe

# Mapeamento reverso para exibir os nomes das classes
# Deve ser o mesmo mapeamento usado no treinamento: 0=negative, 1=neutral, 2=positive
ID_TO_LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Limiar de confiança para classificar como Positivo ou Negativo.
# Se a maior probabilidade não atingir este valor, a previsão tende a ser Neutra.
# Ajuste este valor (entre 0 e 1) conforme sua necessidade.
# Um valor mais alto (ex: 0.8) torna o modelo mais "criterioso" para Positivo/Negativo.
# Um valor mais baixo (ex: 0.5) o torna menos criterioso.
CONFIDENCE_THRESHOLD = 0.7 # Exemplo: 70% de confiança para Positivo ou Negativo

# --- 1. Carregar o Modelo Fine-Tuned ---
# Esta parte do código carrega o modelo BERT e o tokenizador que foram salvos.
try:
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    
    # Verifica se há GPU disponível e move o modelo para ela para inferência mais rápida
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tuned_model.to(device)
    
    # Coloca o modelo em modo de avaliação (importante para inferência)
    fine_tuned_model.eval() 

    print(f"Modelo e tokenizador carregados de '{MODEL_SAVE_PATH}' para inferência.")
except Exception as e:
    print(f"Erro ao carregar o modelo para inferência. Certifique-se de que o modelo foi treinado e salvo corretamente na pasta '{MODEL_SAVE_PATH}'.")
    print(f"Erro: {e}")
    exit()

# --- 2. Exemplos de Manchetes para Classificação ---
print("\n--- Classificando Novas Manchetes ---")
new_headlines = [
    "Lula condena ataque de Israel ao Irã sem criticar resposta de Teerã",
    "Escândalo político abala a credibilidade do congresso.",
    "Nova lei de incentivo à microeconomia é sancionada.",
    "Crise hídrica afeta produção agrícola e gera preocupação.",
    "Presidente participa de cúpula internacional sobre clima.",
    "Reforma tributária avança no parlamento, gerando debates acalorados."
]

# Itera sobre cada manchete para classificá-la
for headline in new_headlines:
    # Tokeniza a manchete: converte texto em IDs numéricos que o BERT entende
    inputs = fine_tuned_tokenizer(
        headline,
        truncation=True, # Trunca se a manchete for muito longa
        padding=True,    # Adiciona preenchimento se for muito curta
        max_length=128,  # Mantém o mesmo max_length usado no treinamento
        return_tensors="pt" # Retorna tensores PyTorch
    ).to(device) # Move os inputs para a GPU/CPU onde o modelo está

    # Faz a previsão sem calcular gradientes (mais rápido para inferência)
    with torch.no_grad():
        outputs = fine_tuned_model(**inputs)

    # Obtém os logits (saídas brutas do modelo) e aplica Softmax para convertê-los em probabilidades
    logits = outputs.logits
    # O [0] pega as probabilidades do primeiro (e único) item no batch
    probabilities = F.softmax(logits, dim=-1)[0] 

    # Extrai a probabilidade para cada classe (Negativo, Neutro, Positivo)
    prob_neg = probabilities[0].item() # Probabilidade da classe 0 (Negativo)
    prob_neutro = probabilities[1].item() # Probabilidade da classe 1 (Neutro)
    prob_pos = probabilities[2].item() # Probabilidade da classe 2 (Positivo)

    # --- Lógica de Pós-processamento de Sentimento ---
    # Começa com Neutro como previsão padrão
    predicted_sentiment = "Neutro" 
    confidence_score = prob_neutro # Começa com a confiança do Neutro

    # Regra 1: Se a probabilidade de Positivo for a maior E atingir o limiar de confiança
    if prob_pos > prob_neg and prob_pos > prob_neutro and prob_pos >= CONFIDENCE_THRESHOLD:
        predicted_sentiment = "Positivo"
        confidence_score = prob_pos
    # Regra 2: Se a probabilidade de Negativo for a maior E atingir o limiar de confiança
    elif prob_neg > prob_pos and prob_neg > prob_neutro and prob_neg >= CONFIDENCE_THRESHOLD:
        predicted_sentiment = "Negativo"
        confidence_score = prob_neg
    # Se nenhuma das regras acima for atendida, mantém como Neutro.
    # A confiança exibida será a da classe com maior probabilidade geral, mesmo que não atinja o threshold,
    # ou a do Neutro se for o caso. Você pode ajustar qual confiança exibir aqui.
    else:
        # Pega o label com a MAIOR probabilidade (mesmo que não atinja o CONFIDENCE_THRESHOLD)
        # e verifica se é Positivo ou Negativo, mas só classifica assim se o threshold for atingido.
        # Caso contrário, cai para Neutro.
        raw_predicted_id = torch.argmax(probabilities).item()
        if (raw_predicted_id == 0 and prob_neg >= CONFIDENCE_THRESHOLD): # É Negativo e confiante
            predicted_sentiment = "Negativo"
            confidence_score = prob_neg
        elif (raw_predicted_id == 2 and prob_pos >= CONFIDENCE_THRESHOLD): # É Positivo e confiante
            predicted_sentiment = "Positivo"
            confidence_score = prob_pos
        else: # Se a maior não foi confiante o suficiente, é Neutro
            predicted_sentiment = "Neutro"
            confidence_score = prob_neutro # Exibe a confiança do Neutro nesse caso

    # Imprime os resultados
    print(f"Manchete: \"{headline}\"")
    print(f"  Probabilidades: Negativo={prob_neg:.4f}, Neutro={prob_neutro:.4f}, Positivo={prob_pos:.4f}")
    print(f"  Sentimento Previsto (Pós-Processado): {predicted_sentiment.capitalize()} (Confiança: {confidence_score:.4f})")
    print("-" * 50)