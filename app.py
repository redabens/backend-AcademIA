from flask import Flask, request,jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Flask est opérationnel avec transformers !"

@app.route('/model', methods=['POST'])
def predict():
    # Récupérer les données JSON envoyées dans le corps de la requête
    data = request.get_json()
    question = data['question']
    context = "Un ensemble est une collection d'objets distincts, appelés éléments, qui partagent une propriété commune. En mathématiques, les ensembles sont fondamentaux car ils servent de base à de nombreuses branches, comme l'arithmétique, l'algèbre et la théorie des probabilités. Les éléments d'un ensemble sont uniques, ce qui signifie qu'un même élément ne peut apparaître qu'une seule fois dans un ensemble. Les ensembles sont souvent notés par des lettres majuscules (A, B, C, ...) et leurs éléments sont listés entre accolades, par exemple A = {1, 2, 3}. Il existe différentes opérations qui permettent de manipuler les ensembles : l'union (fusion des éléments de deux ensembles), l'intersection (éléments communs à deux ensembles), la différence (éléments présents dans un ensemble mais pas dans l'autre) et le complément (éléments de l'univers qui ne sont pas dans un ensemble donné). Un ensemble peut également être infini (par exemple, l'ensemble des nombres naturels) ou vide (ne contenant aucun élément). Les notions de sous-ensemble, d'ensemble universel et de cardinalité permettent d'explorer plus en détail les propriétés des ensembles. La théorie des ensembles joue un rôle clé dans les mathématiques modernes, notamment pour définir les relations, les fonctions et les structures algébriques."
    # Prédiction
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    outputs = model(**inputs)

    # Obtenez les indices du début et de la fin de la réponse
    answer_start = torch.argmax(outputs.start_logits)  # Indice du début de la réponse
    answer_end = torch.argmax(outputs.end_logits) + 1  # Indice de la fin de la réponse (inclusif)

    # Convertissez les indices en mots (tokens)
    answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens)
    return {'answer': answer}, 200

if __name__ == '__main__':
    # Chargement du modèle et du tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    app.run(debug=True, port=3000)