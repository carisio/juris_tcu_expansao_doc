import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Carrega as métricas léxicas e semânticas
with open('metricas-lexicas.pkl', 'rb') as file:
    metricas_lexicas = pickle.load(file)
    
with open('metricas-semanticas.pkl', 'rb') as file:
    metricas_semanticas = pickle.load(file)

# Como mapas não são ordenados, define a ordem para a visualização 
# Obs.: nesse caso, a ordem default acaba coincidindo com o que queremos.
# Mas vamos deixar mais genérico
ordem_expansao_doc = ["BM25.dT5q",
                      "BM25.Syn(GPT3.5)", "BM25.Syn(GPT4o)", "BM25.Syn(Llama3)",
                      "BM25.dT5q.Syn(GPT35)", "BM25.dT5q.Syn(GPT4o)", "BM25.dT5q.Syn(Llama3)"]

ordem_semantico = ["BERT.pt.TCU", "BERT.pt.base", "BERT.pt.legal", "BERT.pt.STJ",
                   "BERT.ml",
                   "OpenAI.small", "OpenAI.large"]

# Faz o mapeamento dos nomes nos gráficos com as propriedades onde estão os resultados
mapeamento_lexico = {
    "Baseline": "bm25_padrao",
    "BM25.dT5q": "doc2query_5queries", 
    "BM25.Syn(GPT3.5)": "sinonimos_enunciado_gpt",
    "BM25.Syn(GPT4o)": "sinonimos_enunciado_gpt_4o",
    "BM25.Syn(Llama3)": "sinonimos_enunciado_llama",
    "BM25.dT5q.Syn(GPT35)": "doc2query_5queries_sinonimos_gpt",
    "BM25.dT5q.Syn(GPT4o)": "doc2query_5queries_sinonimos_gpt_4o",
    "BM25.dT5q.Syn(Llama3)": "doc2query_5queries_sinonimos_llama"
}

mapeamento_semantico = { 
    "BERT.pt.TCU": "bert-base-portuguese-cased-finetuned-tcu-acordaos_mean", 
    "BERT.pt.base": "bert-large-portuguese-cased_mean",
    "BERT.pt.legal": "Legal-BERTimbau-sts-large-ma-v3_mean",
    "BERT.pt.STJ": "bert-large-portuguese-cased-legal-mlm-sts-v1.0_mean",
    "BERT.ml": "paraphrase-multilingual-mpnet-base-v2_mean",
    "OpenAI.small": "text-embedding-3-small_mean",
    "OpenAI.large": "text-embedding-3-large_mean"
}

def plot_metricas(grupo=1):
    # Configuração dos subplots
    label_metricas = ["P@10", "R@10", "MRR@10", "nDCG@10"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Query group: G{grupo}")
    
    mapa_cor = {"Baseline": "lightgray", "expansao_doc": "lightgreen", "semantico": "lightsalmon"}
    
    for idx_metrica, label_metrica in enumerate(label_metricas):      
        ax = axes[idx_metrica // 2, idx_metrica % 2]
        ax.set_ylim(0, 100)
        
        # Gera os dados dos grupos
        # Baseline
        data = [metricas_lexicas[mapeamento_lexico["Baseline"]][label_metrica].tolist()[0 + (grupo-1)*50:50 + (grupo-1)*50]]
        # Métodos léxicos
        for m in ordem_expansao_doc:
            data.append(metricas_lexicas[mapeamento_lexico[m]][label_metrica].tolist()[0 + (grupo-1)*50:50 + (grupo-1)*50])
        # Métodos semânticos
        for m in ordem_semantico:
            data.append(metricas_semanticas[mapeamento_semantico[m]][label_metrica].tolist()[0 + (grupo-1)*50:50 + (grupo-1)*50])
        # Normaliza os valores para %
        data = [[x * 100 for x in sublist] for sublist in data]
    
        colors = [mapa_cor['Baseline']] + [mapa_cor['expansao_doc']]*len(ordem_expansao_doc) + [mapa_cor['semantico']]*len(ordem_semantico)
    
    
        # Criando os boxplots com as cores definidas
        for j in range(len(data)):
            ax.boxplot(data[j], positions=[j+1], widths=0.6, patch_artist=True,
                   medianprops=dict(color='black', linewidth=1),
                   boxprops=dict(facecolor=colors[j]))
            ax.set_xticks(np.arange(1, len(data)+1))
            ax.set_xticklabels(['Baseline'] + ordem_expansao_doc + ordem_semantico, rotation=90)
        ax.set_ylabel(label_metrica)
        
    # Criando os elementos da legenda com as cores corretas
    baseline_patch = mpatches.Patch(color=mapa_cor['Baseline'], label="Baseline")
    expansao_doc_patch = mpatches.Patch(color=mapa_cor['expansao_doc'], label="Document expansion variants")
    semantico_patch = mpatches.Patch(color=mapa_cor['semantico'], label="Semantic variants")
    
    fig.legend(handles=[baseline_patch, expansao_doc_patch, semantico_patch],
           loc="lower center",
           borderpad=1,
           bbox_to_anchor=(0.5, 0.48))
    
    # Ajustando o layout
    plt.tight_layout(rect=[0, 0, 1, 1], h_pad=6, w_pad=5)
    
    # Exibindo a figura
    plt.show()


plot_metricas(3)