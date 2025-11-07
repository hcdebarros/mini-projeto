# 🖼️ Classificação de Imagens com CNN

Projeto de Deep Learning para **classificação de imagens** utilizando uma **Rede Neural Convolucional (CNN)** treinada no dataset **Fashion-MNIST**.  
O objetivo deste projeto é aplicar boas práticas de organização de código em um projeto de Machine Learning, separando o modelo, o dataset e o pipeline de treinamento em arquivos distintos.

---

## 📁 Estrutura do projeto

mini_projeto/
│
├── src/
│ ├── dataset.py # Carrega o dataset Fashion-MNIST e cria os DataLoaders
│ ├── model.py # Define a arquitetura da CNN (SimpleCNN)
│ ├── main.py # Treinamento, validação e visualização no TensorBoard
│
├── data/ # Diretório onde o dataset será baixado automaticamente
├── venv/ # Ambiente virtual (não deve ir para o Git)
├── requirements.txt # Dependências do projeto
└── README.md # Documentação do projeto

yaml
Copiar código

---

## 🚀 Como executar o projeto

### ✅ 1. Criar e ativar a venv

**Windows PowerShell**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
Linux / Mac

bash
Copiar código
python3 -m venv venv
source venv/bin/activate
✅ 2. Instalar dependências
bash
Copiar código
pip install -r requirements.txt
✅ 3. Executar o treinamento
Dentro da pasta do projeto:

bash
Copiar código
python src/main.py
O dataset será baixado automaticamente na pasta data/.

📊 Visualização dos resultados (TensorBoard)
Para acompanhar a acurácia e a loss do treinamento:

bash
Copiar código
tensorboard --logdir runs
Depois acesse no navegador:

👉 http://localhost:6006/

🧠 Modelo utilizado
A CNN está definida no arquivo model.py:

python
Copiar código
self.net = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64*7*7, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 10)
)
Dataset: Fashion-MNIST
Tarefas: Classificação de 10 categorias (camiseta, tênis, bolsa, casaco, etc.)

📦 Tecnologias utilizadas
Python

PyTorch

TorchVision

TensorBoard

NumPy

📌 Objetivo
Estudar boas práticas de organização de projeto em Machine Learning, utilizando uma CNN em PyTorch com TensorBoard para monitoramento.

📝 Licença
Sem licença definida no momento.

✉️ Em caso de dúvidas ou melhorias, fique à vontade para ajustar o código e evoluir o projeto.

yaml
Copiar código

---

Se quiser, posso:

- gerar prints para colocar no README (acurácia e loss no TensorBoard),
- criar um **gif** com o treinamento aparecendo no TensorBoard,
- adicionar uma seção "Resultados".

Quer que eu adicione uma seção de "Resultados" mostrando a accuracy final? 😊
