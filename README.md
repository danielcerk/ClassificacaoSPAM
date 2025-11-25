## Detector de Spam

-----

### Resumo do Projeto

Este projeto é um **Detector de Spam** construído em Python, utilizando técnicas de **Processamento de Linguagem Natural (PLN)** e **aprendizado de máquina** para classificar mensagens de texto (e-mails, SMS, etc.) como **'spam'** ou **'ham'** (não-spam). O modelo é treinado em um *dataset* de mensagens e visa fornecer uma solução de classificação eficiente e de fácil implantação.

-----

### Requisitos

Para executar este projeto, você precisará ter o seguinte instalado em seu sistema:

  * **Python 3.x**

-----

### Começando

Siga estas etapas para clonar o repositório, configurar seu ambiente e executar o projeto.

#### 1\. Clonar o Repositório

Abra seu terminal ou *prompt* de comando e execute o seguinte comando:

```bash
git clone https://github.com/danielcerk/ClassificacaoSPAM.git
cd ClassificacaoSPAM
```

#### 2\. Configurar o Ambiente Virtual

É altamente recomendável usar um **ambiente virtual** para isolar as dependências do projeto.

```bash

python3 -m venv venv
venv\Scripts\activate

```

#### 3\. Instalar as Dependências

Com o ambiente virtual ativado, instale todas as bibliotecas necessárias listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### 4\. Executar o Detector de Spam

```bash

python app.py
```

-----

### Licença

Este projeto está licenciado sob a **Licença MIT** - veja o arquivo [LICENSE.md](LICENSE.md) para mais detalhes.