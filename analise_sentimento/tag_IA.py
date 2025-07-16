import spacy
from pymongo import MongoClient
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

# Conecta ao MongoDB
client = MongoClient(mongo_uri)
db = client["Banco_Coletor"]
colecao = db["Noticias"]

# Carrega spaCy
nlp = spacy.load("pt_core_news_md") # nlp de taamanho medio

# Stopwords adicionais
custom_stopwords = set([
    "fazer", "acontecer", "haver", "dizer", "trazer", "utilizar",
    "ficar", "colocar", "ser", "estar", "ter", "vir", 
    "crescer", "afirmar", "irritar", "puxar", "desenrolar", 
    "mover", "governo", "confusão", "moraes", "ano", "par", "São", "Paulo"
])

# Pré-processamento
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc # lemma = reduz a forma basica. running -> run
        if token.is_alpha and # apenas palavras, sem simbola
        not token.is_stop and # tira palavras do padrão da lib
        token.lemma_ not in custom_stopwords and  # dicionario de rejeição que personalizei
        token.pos_ in {"NOUN", "PROPN"} # substantivos ou pronomes
    ]
    return " ".join(tokens) # retorna tudo numa coisa só(string)

# função para treinar e salvar o modelo
def treinar_e_salvar_modelo(modelo_bertopic="bertopic_tag"):
    documentos_agrupados = []

    for doc in colecao.find():
        manchete = doc.get("manchete", "").strip()
        lide = doc.get("lide", "").strip()
        texto = (manchete + " " + lide).strip()
        if texto:
            documentos_agrupados.append(preprocess(texto))

    vectorizer_model = CountVectorizer(  # vetor de frequencia de vocabulario para o aprendizado
        stop_words=list(nlp.Defaults.stop_words),
        max_df=0.85, # muito frequente. 85%
        min_df=5, # palavras raras. menos que 5 docs diferentes
        token_pattern=r"(?u)\b\w\w+\b"
    )

    topic_model = BERTopic(language="multilingual", vectorizer_model=vectorizer_model)
    topic_model.fit(documentos_agrupados)

    topic_model.save(modelo_bertopic)
    print("sucesso no treinamento")

# função para aplicar modelo salvo em novos documentos
def usar_modelo_em_novos_documentos(modelo_bertopic="bertopic_tag"):
    topic_model = BERTopic.load(modelo_bertopic)

    docs_validos = []
    documentos_agrupados = []

    # Buscar documentos que ainda não têm a tag "assunto"
    for doc in colecao.find({
        "$or": [
            {"assunto": {"$exists": False}},
            {"assunto": {"$size": 0}}
        ]
    }):
        manchete = doc.get("manchete", "").strip()
        lide = doc.get("lide", "").strip()
        texto = (manchete + " " + lide).strip()

        if texto:
            documentos_agrupados.append(preprocess(texto))
            docs_validos.append(doc)

    topics, _ = topic_model.transform(documentos_agrupados)

    def filtrar_palavras_topico(palavras_topico):
        return [
            palavra for palavra, _ in palavras_topico
            if len(palavra) > 2 and palavra not in custom_stopwords
        ]

    atualizados = 0
    docs_atualizados = []
    for doc, topico in zip(docs_validos, topics):
        palavras_topico = topic_model.get_topic(topico)
        palavras_filtradas = filtrar_palavras_topico(palavras_topico)
        tags_formatadas = [{"TAG": palavra.upper()} for palavra in palavras_filtradas]

        # colecao.update_one(
        #     {"_id": doc["_id"]},
        #     {"$set": {
        #         "assunto_IA": 1,
        #         "assunto": tags_formatadas
        #     }}
        # )

        atualizados += 1
        # print(f"Documento {_id} atualizado com: {tags_formatadas}")
        doc["assunto"] = tags_formatadas
        docs_atualizados.append(doc)
        

    print(f"\nTotal de documentos atualizados: {atualizados}")
    
    # 12. Impressão dos documentos atualizados
    print("\nDocumentos atualizados:\n")
    for doc in docs_atualizados:
        print(doc)
        print("-" * 80)

# treinar_e_salvar_modelo()
usar_modelo_em_novos_documentos()
