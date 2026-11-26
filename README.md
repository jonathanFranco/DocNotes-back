# DocNotes — Backend

API do **DocNotes**: envie um PDF e converse com uma IA sobre o conteúdo dele. As respostas saem do próprio documento e indicam a página de cada informação. Tudo roda localmente, sem enviar dados para serviços externos.

Este repositório é a API (FastAPI + Ollama + ChromaDB). A interface web fica no repositório **frontend** do projeto.

## Como funciona

O DocNotes usa **RAG** (*Retrieval-Augmented Generation*): em vez de pedir para a IA "lembrar" do documento, a API busca os trechos mais relevantes para cada pergunta e entrega só esses trechos ao modelo, que responde com base neles.

```
Upload:   PDF → extração por página → limpeza → chunks → embeddings → ChromaDB
Pergunta: pergunta → embedding → 4 trechos mais parecidos → filtro de relevância → LLM → resposta + páginas
```

## Stack

| Camada | Tecnologia |
|---|---|
| API | Python 3.12, FastAPI, Pydantic |
| IA local | [Ollama](https://ollama.com) com `qwen2.5:3b` (respostas) e `embeddinggemma` (embeddings) |
| PDF | PyMuPDF |
| Banco vetorial | ChromaDB (persistido em disco) |

## Início rápido com Docker (recomendado)

Só precisa do **Docker** com o **Docker Compose v2**. Não é preciso instalar Python nem Ollama.

```bash
docker compose up --build
```

O compose sobe três serviços:

1. **`ollama`**: o Ollama, em container.
2. **`ollama-models`**: baixa os modelos `qwen2.5:3b` e `embeddinggemma` e encerra.
3. **`backend`**: a API, que só sobe depois que os modelos estiverem prontos.

> **Primeira subida:** o download dos modelos (~2,5 GB) pode levar alguns minutos. Eles ficam salvos no volume `ollama`, então nas próximas vezes a subida é instantânea.

Quando aparecer `Uvicorn running on http://0.0.0.0:8000`, a API está pronta:

- API: http://localhost:8000
- Documentação interativa (Swagger): http://localhost:8000/docs
- Status: http://localhost:8000/health

### Com GPU NVIDIA (opcional)

Por padrão o Ollama roda na CPU, o que funciona em qualquer máquina. Se você tem GPU NVIDIA com o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) instalado, as respostas ficam bem mais rápidas assim:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

O `docker-compose.gpu.yml` só libera a GPU para o serviço `ollama`. Sem o toolkit, esse comando falha. Nesse caso, use o comando padrão.

### Comandos úteis

```bash
docker compose up -d --build      # sobe em segundo plano
docker compose logs -f backend    # acompanha os logs da API
docker compose down               # para tudo (mantém modelos e documento)
docker compose down -v            # para e APAGA os volumes: modelos baixados e documento indexado
```

### Observações

- **Hot reload:** a pasta `app/` é montada no container. Ao editar o código, a API reinicia sozinha.
- **Ollama isolado:** o Ollama do container não publica a porta 11434, então não conflita com um Ollama instalado na sua máquina.
- **Porta 8000:** a API em Docker e uma API rodando direto na máquina (sem Docker) disputam a mesma porta. Rode uma de cada vez.
- **Dados separados:** em Docker, o ChromaDB fica no volume `chroma`. Sem Docker, fica em `./chroma_db`. Um documento enviado num modo não aparece no outro.

## Rodando sem Docker

Útil para desenvolvimento, usando o Ollama instalado na sua máquina.

**Pré-requisitos:** Python ≥ 3.12 e Ollama ≥ 0.11.10 (versão mínima para o `embeddinggemma`).

```bash
# 1. Ollama e modelos
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
ollama pull embeddinggemma

# 2. Dependências
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Configuração
cp .env.example .env

# 4. API em http://localhost:8000
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Se o frontend for rodar em Docker e a API direto na máquina, suba o uvicorn com `--host 0.0.0.0`. Assim o container do frontend consegue alcançar a API.

## Configuração

Toda a configuração fica em [app/config.py](app/config.py). Os valores vêm de variáveis de ambiente ou do arquivo `.env` (modelo em [.env.example](.env.example)).

| Variável | Padrão | Descrição |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | URL do Ollama |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Timeout das chamadas ao Ollama |
| `LLM_MODEL` | `qwen2.5:3b` | Modelo que gera as respostas |
| `EMBEDDING_MODEL` | `embeddinggemma` | Modelo de embeddings |
| `CHROMA_DIR` | `chroma_db` | Pasta do ChromaDB (caminho relativo à raiz do backend) |
| `MAX_UPLOAD_MB` | `20` | Tamanho máximo do PDF |

Com Docker, `OLLAMA_HOST` e `CHROMA_DIR` são definidos pelo `docker-compose.yml`. Para trocar os modelos, use variáveis de ambiente, e os novos modelos são baixados na subida:

```bash
LLM_MODEL=llama3.2:3b docker compose up
```

## API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status da API, do Ollama e dos modelos |
| GET | `/documents` | Dados do documento atual (404 se não houver) |
| POST | `/documents/upload` | Envia um PDF (`multipart/form-data`, campo `file`), processa e substitui o documento atual |
| DELETE | `/documents` | Exclui o documento atual e seus vetores |
| POST | `/chat` | Recebe `{"question": "..."}` e responde com base no documento atual |

A API trabalha com **um documento por vez**: um novo upload substitui o anterior.

### Exemplos

```bash
curl localhost:8000/health
curl -F "file=@contrato.pdf" localhost:8000/documents/upload
curl -H "Content-Type: application/json" -d '{"question": "Qual o prazo de entrega?"}' localhost:8000/chat
curl -X DELETE localhost:8000/documents
```

Resposta do `/chat`:

```json
{
  "answer": "O prazo de entrega é de 15 dias úteis (p. 3).",
  "found": true,
  "sources": [{ "page": 3, "excerpt": "…trecho usado…", "similarity": 0.61 }]
}
```

As respostas são sempre em português: o prompt, também em português, instrui o modelo a isso. Quando o documento não contém a resposta, `found` vale `false`, `answer` vem vazio e `sources` também: a mensagem de "não encontrei" fica a cargo de quem consome a API (o frontend mostra "Não encontrei essa informação no documento.").

No `/health`, `status` vale `ok` quando o Ollama responde e os dois modelos estão instalados. Nos outros casos vale `degraded`, e o campo `ollama.error` explica o problema.

### Erros

Os erros retornam um `code` estável, uma mensagem em inglês em `detail` e, quando relevante, valores em `params`:

```json
{ "code": "FILE_TOO_LARGE", "detail": "The file exceeds the 20 MB limit.", "params": { "max_mb": 20 } }
```

O `detail` serve para logs e para quem consome a API. Interfaces devem montar a mensagem a partir do `code`: o frontend faz isso em português.

| Status | `code` | Situação | `params` |
|---|---|---|---|
| 400 | `FILE_NOT_PDF` | Arquivo sem extensão `.pdf` | |
| 400 | `FILE_EMPTY` | Arquivo vazio | |
| 404 | `NO_DOCUMENT` | `GET`/`DELETE /documents` sem documento carregado | |
| 409 | `NO_DOCUMENT` | `/chat` sem documento carregado | |
| 413 | `FILE_TOO_LARGE` | Arquivo maior que `MAX_UPLOAD_MB` | `max_mb` |
| 422 | `PDF_INVALID` | O arquivo não é um PDF | |
| 422 | `PDF_CORRUPTED` | PDF corrompido | |
| 422 | `PDF_PASSWORD_PROTECTED` | PDF protegido por senha | |
| 422 | `PDF_NO_PAGES` | PDF sem páginas | |
| 422 | `PDF_NO_TEXT` | PDF sem texto extraível (digitalizado) | |
| 422 | `QUESTION_EMPTY` | Pergunta vazia | |
| 422 | `QUESTION_TOO_LONG` | Pergunta com mais de 1000 caracteres | `max_chars` |
| 503 | `OLLAMA_UNAVAILABLE` | Ollama fora do ar | |
| 503 | `OLLAMA_MODEL_ERROR` | Ollama recusou o modelo (por exemplo, não instalado) | `model` |

Erros de validação do próprio FastAPI (por exemplo, corpo da requisição sem o campo `question`) seguem o formato padrão do framework, sem `code`.

## Por dentro

### Processamento do PDF

- **Extração:** texto página a página com PyMuPDF.
- **Limpeza:** normalização Unicode (NFKC), remoção de caracteres de controle, junção de palavras hifenizadas e de linhas do mesmo parágrafo.
- **Chunking:** cada página é dividida por sentenças em blocos de até ~1000 caracteres, com ~150 de sobreposição. Todo trecho tem uma página de origem exata.
- **Embeddings:** gerados em lotes, com o prefixo de tarefa recomendado pelo `embeddinggemma`.
- **Troca segura:** o novo documento só substitui o atual depois que todo o processamento dá certo. Se o Ollama falhar no meio, o documento anterior continua salvo.

### Respostas (RAG)

- **Busca:** os 4 trechos mais parecidos com a pergunta, por similaridade de cosseno.
- **Relevância:** trechos com similaridade abaixo de 0,20 são descartados. Se nenhum sobrar, a API responde "não encontrei" sem chamar a LLM.
- **Reprodutível:** `temperature 0` e seed fixa, então a mesma pergunta gera a mesma resposta.
- **Fontes:** a resposta lista só os trechos das páginas que cita no formato `(p. N)`.

### Guardrails

| Camada | O que faz |
|---|---|
| Pergunta | Rejeita perguntas vazias ou com mais de 1000 caracteres |
| Contexto | Limita a ~3000 caracteres e descarta trechos irrelevantes |
| Prompt | Instrui a responder só com base nos trechos, citar páginas e tratar o documento como dado, não como instrução. Repete as regras no fim, porque modelos pequenos dão mais peso ao final do prompt |
| Sanitização | Remove do texto enviado à LLM frases com padrão de *prompt injection* ("ignore as instruções…") e tags que quebrariam a delimitação do documento |
| Saída | Detecta respostas "não encontrei", remove citações duplicadas e, se o modelo se contradiz, pede uma releitura do trecho citado |

### Limitações conhecidas

- O `qwen2.5:3b` é conservador: em perguntas que exigem inferência, ele pode responder "não encontrei" mesmo quando a informação está implícita no texto. Num teste com 36 perguntas sobre 3 PDFs, acertou 35 e **não inventou nenhuma resposta** para perguntas fora do documento.
- Ele acha o trecho certo, mas pode explicar errado conteúdos que pedem raciocínio, como matemática passo a passo. Confira sempre as fontes.
- PDFs digitalizados (só imagem) não são suportados, porque não há OCR.
- A primeira pergunta depois de um tempo sem uso pode demorar alguns segundos, enquanto o Ollama carrega o modelo na memória.

## Estrutura

```
app/
  main.py                   app FastAPI e registro das rotas
  config.py                 configuração (variáveis de ambiente / .env)
  errors.py                 AppError e handler: erros com `code` estável
  routes/                   health.py, documents.py, chat.py
  schemas/                  contratos Pydantic de entrada e saída
  services/
    pdf_processor.py        extração, limpeza e chunking
    embedding_service.py    embeddings via Ollama
    vector_store.py         ChromaDB (um documento por vez)
    rag_service.py          pipeline RAG
    guardrails.py           validação, contexto, prompt e sanitização
Dockerfile                  imagem da API
docker-compose.yml          Ollama + download dos modelos + API
docker-compose.gpu.yml      override opcional para GPU NVIDIA
```
