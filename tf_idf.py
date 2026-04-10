import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "O detetive mais famoso do mundo usa suas célebres habilidades de dedução neste livro, em que o Dr. John Watson é apresentado. Recentemente dispensado do serviço militar, Watson ocupa um quarto com um jovem incrível: o arrogante especialista em investigação criminal, Sherlock Holmes. E o acontecimento de um crime bizarro prova ser um começo promissor de uma das mais ilustres parcerias de todos os tempos.",
    "O DETETIVE MAIS FAMOSO DE TODOS OS TEMPOS, Sherlock Holmes, ao lado do Dr. Watson, investiga a misteriosa morte, em um pântano, de Sir Charles Baskerville, nesta que é uma das mais populares histórias do detetive, adaptada várias vezes para o cinema e a TV. No centro do caso, um suposto cão assassino e sobrenatural, que assombra a família Baskerville, cujo último herdeiro teme pela vida, depois de receber um enigmático bilhete. Com muitas pistas desencontradas e uma aterradora mansão com mais de 500 anos, Holmes precisa se valer de toda a sua argúcia para identificar os verdadeiros culpados.",
    "Espirituoso, charmoso, brilhante, astuto e possivelmente o maior ladrão do mundo, Lupin se depara com o único homem que pode ser capaz de detê-lo: nada menos do que o grande cavalheiro-detetive britânico Herlock Sholmes! Quem sairá triunfante? Este clássico contém duas histórias que não permitem ao leitor tirar os olhos das páginas do livro: Em A mulher loura, dividida em seis capítulos, dois acontecimentos envolvendo uma dama loura e desaparecimentos misteriosos agitam a vida de Lupin. Em A lâmpada judaica, dois capítulos eletrizantes e divertidos, uma joia preciosa desaparece e Herlock Sholmes é chamado para desvendar o mistério. ",
    "Com 'Os crimes da rua Morgue', Edgar Allan Poe inaugurou, em 1841, a moderna literatura policial e criou um de seus mais célebres detetives, o até hoje reverenciado Auguste Dupin. O conto, que narra a memorável investigação do assassinato de duas mulheres em um quarto fechado, é o carro-chefe desta reunião de histórias de terror e mistério traduzida por ninguém menos que Clarice Lispector. Grande leitora e fã da literatura policial, a escritora, que também verteu para o português os livros de Agatha Christie sob o pseudônimo de Mary Westmacott, empresta seu talento invulgar ao gênio de Poe, trazendo para o leitor brasileiro histórias como 'A máscara da morte rubra', 'O gato preto', 'Ligeia' e outras. Lançamento do selo Fantástica Rocco, esta edição de Os crimes da rua Morgue e outras histórias extraordinárias recupera este encontro, literalmente, fantástico.",
    "Bilbo Bolseiro era um dos mais respeitáveis hobbits de todo o Condado até que, um dia, o mago Gandalf bate à sua porta. A partir de então, toda sua vida pacata e campestre soprando anéis de fumaça com seu belo cachimbo começa a mudar. Ele é convocado a participar de uma aventura por ninguém menos do que Thorin Escudo-de-Carvalho, um príncipe do poderoso povo dos Anãos. Esta jornada fará Bilbo, Gandalf e 13 anãos atravessarem a Terra-média, passando por inúmeros perigos, como os imensos trols, as Montanhas Nevoentas infestadas de gobelins ou a muito antiga e misteriosa Trevamata, até chegarem (se conseguirem) na Montanha Solitária. Lá está um incalculável tesouro, mas há um porém. Deitado em cima dele está Smaug, o Dourado, um dragão malicioso que... bem, você terá que ler para descobrir.",
]

stopwords = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas",
    "para", "ao", "aos", "à", "às","com", "sem", "e", "ou", "mas", "que", "se", "como", "até", 
    "nem", "porém", "ele", "ela", "eles", "elas", "seu", "sua", "seus", "suas", "este", "esta", 
    "estes", "estas", "esse", "essa", "esses", "essas", "neste", "nesta", "nestes", "nestas",
    "isso", "isto", "aqui", "é", "são", "ser", "está", "era", "foi", "há", "tem", "ter",
    "não", "mais", "muito", "também", "já", "bem", "toda", "todo", "todos", "todas",
    "menos", "apenas", "nada", "depois",
}

def preprocessamento(text: str) -> str:
    text = text.lower()
    text = text.replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)

docs_processado = [preprocessamento(doc) for doc in documents]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs_processado)
vocabulario = vectorizer.get_feature_names_out()
matrix_dense = tfidf_matrix.toarray().T
header = "\nTermo".ljust(18) + "".join(f"Doc{i}".rjust(8) for i in range(len(documents)))
query = "detetive Sherlock Holmes"
query_processada = preprocessamento(query)
query_vector = vectorizer.transform([query_processada])
pontuacao = cosine_similarity(query_vector, tfidf_matrix).flatten()
ranking = np.argsort(pontuacao)[::-1]

print("Documentos Pré-Processados:\n")
for i, doc in enumerate(docs_processado):
    print(f"Doc {i}: {doc}\n")
print("\nVocabulario:")
print(f" {list(vocabulario)}")
print(f" {header}")
print(f" {'-' * len(header)}")
for j, term in enumerate(vocabulario):
    coluna = term.ljust(18) + "".join(f"{matrix_dense[j, i]:8.4f}" for i in range(len(documents)))
    print(f"  {coluna}")
print(f"\nConsulta: \"{query}\"")
print("\nVetor da consulta (TF-IDF):")
for i, val in enumerate(query_vector.toarray()[0]):
    if val > 0:
        print(f" {vocabulario[i]}: {val:.4f}")
print("Pontuação de Similaridade (cosseno): ")
for i, pont in enumerate(pontuacao):
    print(f"  Doc {i} (pontuacao = {pont:.4f})")
print("\nRANKING FINAL (por relevância)")
for pos, i in enumerate(ranking, start=1):
    print(f" {pos}º — Doc {i} (pontuacao = {pontuacao[i]:.4f})")
print("\nDOCUMENTO MAIS RELEVANTE\n")
best = ranking[0]
print(f"  Doc {best}: \"{documents[best]}\"")
print(f"  pontuacao: {pontuacao[best]:.4f}")