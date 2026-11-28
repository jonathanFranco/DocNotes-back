# DocNotes — Backend

API do **DocNotes**: envie um PDF e receba um resumo ponto a ponto, na ordem do documento, com a página de cada informação. Cada ponto é conferido no próprio PDF com RAG e vem com o trecho de origem. Tudo roda localmente, sem enviar dados para serviços externos.

Este repositório é a API (FastAPI + Ollama + ChromaDB). A interface web fica no repositório **frontend** do projeto.

## Como funciona

Um PDF raramente cabe de uma vez no contexto do modelo. Por isso o resumo é feito em duas etapas (*map-reduce*), num processo em segundo plano que começa sozinho logo após o upload:

```
Upload:     PDF → extração por página → limpeza → chunks → embeddings → ChromaDB → resumo começa em segundo plano
Map:        páginas agrupadas em blocos de ~2000 caracteres → para cada bloco, o modelo lista os pontos de cada página
Reduce:     anotações juntadas na ordem do documento → resumo final, transmitido enquanto é escrito
Verificação (RAG): para cada ponto → busca dos trechos mais parecidos no ChromaDB → checagem de números + juiz → confirmado ou não
```

O frontend consulta `GET /documents/summary` a cada segundo e mostra o progresso da leitura, os pontos à medida que o modelo os escreve e o resultado da conferência de cada um.

## Stack

| Camada | Tecnologia |
|---|---|
| API | Python 3.12, FastAPI, Pydantic |
| IA local | [Ollama](https://ollama.com) com `qwen2.5:3b` (resumo e verificação) e `embeddinggemma` (embeddings) |
| PDF | PyMuPDF |
| Banco vetorial | ChromaDB (trechos do documento, para a verificação) |
| Armazenamento | Um arquivo JSON com o documento e o resumo |

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

Por padrão o Ollama roda na CPU, o que funciona em qualquer máquina. Se você tem GPU NVIDIA com o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) instalado, os resumos ficam bem mais rápidos assim:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

O `docker-compose.gpu.yml` só libera a GPU para o serviço `ollama`. Sem o toolkit, esse comando falha. Nesse caso, use o comando padrão.

### Comandos úteis

```bash
docker compose up -d --build      # sobe em segundo plano
docker compose logs -f backend    # acompanha os logs da API
docker compose down               # para tudo (mantém modelos, documento e resumo)
docker compose down -v            # para e APAGA os volumes: modelos baixados, documento e resumo
```

### Observações

- **Hot reload:** a pasta `app/` é montada no container. Ao editar o código, a API reinicia sozinha. Um resumo em andamento é perdido no reinício, e recomeça na próxima consulta do frontend.
- **Ollama isolado:** o Ollama do container não publica a porta 11434, então não conflita com um Ollama instalado na sua máquina.
- **Porta 8000:** a API em Docker e uma API rodando direto na máquina (sem Docker) disputam a mesma porta. Rode uma de cada vez.
- **Dados separados:** em Docker, o documento e o índice do ChromaDB ficam no volume `data`. Sem Docker, ficam em `./data`. Um documento enviado num modo não aparece no outro.

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
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Timeout de cada chamada ao Ollama |
| `LLM_MODEL` | `qwen2.5:3b` | Modelo que escreve e confere o resumo |
| `EMBEDDING_MODEL` | `embeddinggemma` | Modelo de embeddings usado na busca |
| `DATA_DIR` | `data` | Pasta do documento, do resumo e do índice do ChromaDB (caminho relativo à raiz do backend) |
| `MAX_UPLOAD_MB` | `20` | Tamanho máximo do PDF |
| `LOG_LEVEL` | `INFO` | Nível de log; cada chamada ao modelo gera uma linha com tokens, limite e duração |

Com Docker, `OLLAMA_HOST` e `DATA_DIR` são definidos pelo `docker-compose.yml`. Para trocar os modelos, use as variáveis de ambiente, e os novos modelos são baixados na subida:

```bash
LLM_MODEL=llama3.2:3b docker compose up
```

## API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status da API, do Ollama e dos dois modelos |
| GET | `/documents` | Dados do documento atual (404 se não houver) |
| POST | `/documents/upload` | Envia um PDF (`multipart/form-data`, campo `file`), indexa os trechos, substitui o documento atual e começa o resumo |
| GET | `/documents/summary` | Estado do resumo. Consulte de novo enquanto `status` for `generating` |
| POST | `/documents/summary` | Recomeça o resumo depois de uma falha (não faz nada se ele estiver pronto ou em andamento) |
| DELETE | `/documents` | Exclui o documento, o resumo e os vetores, e interrompe um resumo em andamento |

A API trabalha com **um documento por vez**: um novo upload substitui o anterior e cancela o resumo que estava sendo gerado.

### Exemplos

```bash
curl localhost:8000/health
curl -F "file=@contrato.pdf" localhost:8000/documents/upload
curl localhost:8000/documents/summary
curl -X DELETE localhost:8000/documents
```

Resposta do `/documents/summary` pronta:

```json
{
  "status": "ready",
  "stage": null,
  "progress": null,
  "points": [
    {
      "text": "O contrato tem vigência de 24 meses, prorrogável até 60 meses (p. 2).",
      "verified": true,
      "evidence": { "page": 2, "excerpt": "O contrato terá vigência de 24 meses, contados a partir de 1º de março de 2026…" }
    }
  ],
  "error": null
}
```

- `stage` mostra a fase: `reading` enquanto o modelo lê os blocos (`progress` diz quantos já foram lidos), `writing` enquanto o resumo final é escrito (os pontos chegam aos poucos e o último pode estar incompleto) e `checking` enquanto cada ponto é conferido (`progress` conta os pontos).
- Cada ponto termina com as páginas citadas pelo modelo, no formato `(p. N)`.
- `verified` vale `true` quando o documento confirma o ponto, `false` quando não confirma e `null` enquanto o ponto ainda não foi conferido. `evidence` traz o trecho do PDF mais parecido com o ponto e a página dele, que pode diferir da página citada quando o modelo citou errado.
- Se a geração falhar, `status` vale `failed` e `error` traz o `code` (veja abaixo). `POST /documents/summary` tenta de novo.
- O resumo pronto fica salvo: as próximas consultas respondem na hora, mesmo depois de reiniciar a API.

No `/health`, `status` vale `ok` quando o Ollama responde e o modelo está instalado. Nos outros casos vale `degraded`, e o campo `ollama.error` explica o problema.

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
| 404 | `NO_DOCUMENT` | Nenhum documento carregado | |
| 413 | `FILE_TOO_LARGE` | Arquivo maior que `MAX_UPLOAD_MB` | `max_mb` |
| 422 | `PDF_INVALID` | O arquivo não é um PDF | |
| 422 | `PDF_CORRUPTED` | PDF corrompido | |
| 422 | `PDF_PASSWORD_PROTECTED` | PDF protegido por senha | |
| 422 | `PDF_NO_PAGES` | PDF sem páginas | |
| 422 | `PDF_NO_TEXT` | PDF sem texto extraível (digitalizado) | |

Erros que acontecem durante a geração do resumo não viram status HTTP: aparecem em `error.code` na resposta do `/documents/summary`.

| `code` | Situação | `params` |
|---|---|---|
| `OLLAMA_UNAVAILABLE` | Ollama fora do ar | |
| `OLLAMA_MODEL_ERROR` | Ollama recusou o modelo (por exemplo, não instalado) | `model` |
| `SUMMARY_EMPTY` | O modelo não devolveu nenhum ponto | |
| `SUMMARY_FAILED` | Erro inesperado (detalhes no log da API) | |

## Por dentro

### Processamento do PDF

- **Extração:** texto página a página com PyMuPDF.
- **Limpeza:** normalização Unicode (NFKC), remoção de caracteres de controle, junção de palavras hifenizadas e de linhas do mesmo parágrafo.
- **Armazenamento:** o documento (texto por página) e, depois, o resumo ficam num único `document.json`, gravado por escrita atômica (arquivo temporário + renomeação).

### Resumo (map-reduce)

- **Blocos:** páginas consecutivas são agrupadas em blocos de até ~2000 caracteres. Uma página maior que isso é dividida entre sentenças.
- **Map:** para cada bloco, o modelo lista de 1 a 5 pontos por página, sob títulos `Página N:`. As páginas das citações vêm desses títulos e quem as escreve é o código, não o modelo. Nos testes, o modelo errava ou omitia citações no meio do texto, mas seguia a estrutura de títulos.
- **Reduce:** as anotações vão, na ordem, para uma última chamada que junta pontos repetidos e escreve o resumo final, com até 15 pontos, mantendo as páginas. Se as anotações não couberem numa chamada, elas são condensadas em rodadas antes.
- **Sempre as duas etapas:** mesmo um documento curto passa pelo reduce. Nos testes, usar direto a lista por página gerava pontos repetidos e estourava o limite de tokens.
- **Streaming:** o resumo final é lido do Ollama aos pedaços. A cada consulta, a API devolve os pontos já escritos.
- **Reprodutível:** `temperature 0` e seed fixa.
- **Tempo:** num PDF de 6 páginas, com o Ollama numa RTX 3050 de 4 GB, o resumo leva ~20 s. Cada bloco a mais soma alguns segundos.

### Verificação com RAG

O resumo é escrito por um modelo de 3B, que às vezes distorce uma informação. Por isso cada ponto passa por uma conferência no próprio documento:

1. **Indexação:** no upload, cada página é dividida em trechos de até ~1000 caracteres, com ~150 de sobreposição. Os trechos viram vetores (`embeddinggemma`) e vão para o ChromaDB.
2. **Busca:** cada ponto, sem as citações, vira um vetor de consulta. A busca traz os 3 trechos mais parecidos, e o sistema junta a eles os trechos das páginas que o ponto cita (até ~3500 caracteres).
3. **Números:** todo número do ponto (valores, prazos, percentuais, datas) precisa aparecer nesses trechos. É uma checagem determinística.
4. **Juiz:** o `qwen2.5:3b` recebe os trechos e o ponto e responde só SIM ou NÃO para "o trecho sustenta tudo o que a afirmação diz?".
5. **Resultado:** o ponto é confirmado só se passar nas duas checagens. Um ponto não confirmado não é apagado: aparece marcado, com o trecho mais próximo ao lado.

A similaridade de embeddings sozinha não serve como filtro. Na calibração, pontos verdadeiros ficaram entre 0,43 e 0,77 de similaridade com o melhor trecho, e pontos com um número trocado ficaram entre 0,54 e 0,64, porque falam do mesmo assunto. Já a combinação de números e juiz, num teste com 16 afirmações, aceitou 6 de 7 verdadeiras e barrou as 9 falsas, inclusive a troca de "4 h para urgentes / 48 h para os demais" por "4 h para preventiva / 48 h para corretiva". O único erro foi barrar um ponto verdadeiro cuja informação estava num trecho que a busca não trouxe. Por isso os trechos das páginas citadas entram junto. A conferência leva ~0,3 s por ponto.

### Guardrails

| Camada | O que faz |
|---|---|
| Sanitização | Remove do texto enviado ao modelo frases com padrão de *prompt injection* ("ignore as instruções…"), em português e inglês, e tags que quebrariam a delimitação do documento. Frases dirigidas à IA ("IA: …") só contam como injeção no início da linha, para não derrubar linhas como "Habilidades: Python, AI, ML" |
| Prompt | Instrui a usar só o texto, não opinar nem inventar, preservar números, datas e nomes, e tratar o documento como dado, não como instrução. Os prompts são em português, porque o resumo é em português |
| Lembrete final | Repete o formato no fim do prompt, porque modelos pequenos dão mais peso ao final |
| Saída | Aproveita só os itens de lista, ignora introduções, remove pontos repetidos, normaliza citações para `(p. N)` e descarta páginas que não existem no documento |
| Verificação | Cada ponto é conferido no documento com RAG, checagem de números e juiz (veja acima) |

### Limitações conhecidas

- Um modelo de 3B resume bem os fatos principais, mas pode juntar duas informações de um jeito errado. A verificação marca esses casos, mas não é infalível: ela pode deixar passar um erro sutil, e também é rigorosa, podendo marcar como não confirmada uma simplificação correta. Confira os trechos.
- Detalhes secundários podem ficar de fora no reduce, que limita o resumo a 15 pontos.
- PDFs digitalizados (só imagem) não são suportados, porque não há OCR.
- O primeiro resumo depois de um tempo sem uso demora alguns segundos a mais, enquanto o Ollama carrega o modelo na memória.

## Estrutura

```
app/
  main.py                   app FastAPI e registro das rotas
  config.py                 configuração (variáveis de ambiente / .env)
  errors.py                 AppError e handler: erros com `code` estável
  routes/                   health.py, documents.py
  schemas/                  contratos Pydantic de entrada e saída
  services/
    pdf_processor.py        extração, limpeza e divisão do texto
    document_store.py       documento e resumo em JSON (um documento por vez)
    embedding_service.py    embeddings via Ollama
    vector_store.py         ChromaDB: trechos do documento e busca por similaridade
    summary_service.py      resumo map-reduce em segundo plano
    rag_service.py          RAG: conferência de cada ponto do resumo no documento
    llm.py                  chamadas ao Ollama
    guardrails.py           prompts, sanitização e leitura da resposta do modelo
Dockerfile                  imagem da API
docker-compose.yml          Ollama + download dos modelos + API
docker-compose.gpu.yml      override opcional para GPU NVIDIA
```
