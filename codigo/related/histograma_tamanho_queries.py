import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

PASTA_JURIS_TCU = '../src/dados/outputs/1_tratamento_juris_tcu/'
queries = pd.read_csv(f'{PASTA_JURIS_TCU}query_tratado.csv', sep='|')


def plot_histogram(array1, array2, array3, labels, nome_figura_salvar = None):
    # Contar a quantidade de palavras em cada string
    word_counts1 = [len(s.split()) for s in array1]
    word_counts2 = [len(s.split()) for s in array2]
    word_counts3 = [len(s.split()) for s in array3]
    
    # Calcular as médias
    mean1 = np.mean(word_counts1)
    mean2 = np.mean(word_counts2)
    mean3 = np.mean(word_counts3)
    
    # Contar a frequência de cada quantidade de palavras
    counter1 = Counter(word_counts1)
    counter2 = Counter(word_counts2)
    counter3 = Counter(word_counts3)
    
    # Obter todas as quantidades únicas de palavras
    all_keys = sorted(set(counter1.keys()) | set(counter2.keys()) | set(counter3.keys()))
    key_index = {key: i for i, key in enumerate(all_keys)}  # Mapear valores reais para os índices do eixo X
    
    # Criar listas de frequências
    freq1 = [counter1[key] for key in all_keys]
    freq2 = [counter2[key] for key in all_keys]
    freq3 = [counter3[key] for key in all_keys]
    
    x = np.arange(len(all_keys))  # Posições no eixo X
    width = 0.25  # Largura das barras
    
    # Criar o gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, freq1, width, label=labels[0], alpha=0.7)
    ax.bar(x, freq2, width, label=labels[1], alpha=0.7)
    ax.bar(x + width, freq3, width, label=labels[2], alpha=0.7)

    # Adicionar linhas de grade horizontais tracejadas
    ax.yaxis.grid(True, linestyle='dashed', alpha=0.5)
    
    # Plotar as médias corretamente sem arredondamento
    ax.axvline(np.interp(mean1, all_keys, x), color='tab:blue', linestyle='dashed', linewidth=1, label=f'Mean {labels[0]}')
    ax.axvline(np.interp(mean2, all_keys, x), color='tab:orange', linestyle='dashed', linewidth=1, label=f'Mean {labels[1]}')
    ax.axvline(np.interp(mean3, all_keys, x), color='tab:green', linestyle='dashed', linewidth=1, label=f'Mean {labels[2]}')
    
    # Ajustar rótulos e legenda
    ax.set_xlabel("Number of words per query", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    if nome_figura_salvar is not None:
        plt.savefig(nome_figura_salvar, dpi=300, bbox_inches='tight')
    plt.show()

# Exemplo de uso
queries_list = queries.TEXT.tolist()
g1 = queries_list[0:50]
g2 = queries_list[50:100]
g3 = queries_list[100:150]
labels = ["G1", "G2", "G3"]
plot_histogram(g1, g2, g3, labels, "histogram_words_query.png")