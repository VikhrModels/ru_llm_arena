# Русский SBS бенчмарк на основе кода Arena-Hard-Auto

### Описание
Это инструмент для **автоматической** оценки моделей на **русском языке** с помощью **сильной LLM** **(GPT-4-1106-preview)**. Использует систему [ELO рангов](https://ru.wikipedia.org/wiki/Рейтинг_Эло).

Основывается на фиксированном наборе из 500 промптов, разбитым по 50 темам. Каждая модель дает свой ответ на каждый промпт,
после чего он сравнивается с ответами на эти же промпты от **модели-бейзлайна (gpt-3.5-turbo-0125)**.

Важными особенностями отличающими Arena-Hard-Auto от обычного SBS ялвются:
- При сравнениях ответов учитываются 3 основных случая: >> (сильно лучше), > (просто лучше) и = (примерно одинаково), за случаи когда один ответ сильно лучше другого вес вердикта увеличивается в 3 раза
- Для удаления позиционного биаса в промпте модели-судьи, каждое сравнение делается 2 раза (ответы моделей переставляются местами в промпте).
- Бутстрапирование результатов сравнений для получения доверительных интервалов
- Использование системы ELO рангов и предсказания винрейта с помощью [модели Bradley–Terry](https://en.wikipedia.org/wiki/Bradley–Terry_model)

В отличие от [оригинала Arena-Hard-Auto](https://github.com/lm-sys/arena-hard-auto), этот репозиторий содержит некоторые изменения:
1. Изменен промпт для модели-оценщика, для того чтобы сравнивать модели в том числе по владению русским языком, сам промпт находится в `config/judge_config.yaml`
2. Добавлена функция контроля длины ответов для штрафования за слишком длинные ответы по сравнению с бейзлайном (экспериментально)
3. В качестве бейзлайна используется gpt-3.5-turbo-0125, в отличие от GPT-4, так как для русского языка модели менее развиты чем для английского
4. Добавлены функции генерации с gigachat и yandexgpt
5. Фиксы некоторых багов в оригинальной имплементации
6. Использование быстрой реализации алгоритма расчёта рангов из пакета [Evalica](https://github.com/dustalov/evalica)

### Датасет с промптами
Для этой арены существует 2 датасета, но используется сейчас только первый:
1. [General](https://huggingface.co/datasets/Vikhrmodels/ru-arena-general) (Диверсифицированные по 50 топикам вопросы из онлайн lmsys арены) - Именно он используется сейчас
2. [Hard](https://huggingface.co/datasets/Vikhrmodels/ru-arena-hard) (Переведенный оригинальный датасет из английской Arena-Hard)

## Состояние арены на 27.10.2024

На текущий момент в рейтинге находятся *43* модели \
`score` - предсказаный винрейт модели относительно бейзлайна (gpt-3.5-turbo-0125)

### Без контроля длины
```console
> python show_result.py
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Model                                    ┃ Score ┃      95% CI ┃ Avg. #Tokens ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ gpt-4-1106-preview                       │  90.9 │ (-1.3, 1.2) │          541 │
│ gpt-4o-mini                              │  83.9 │ (-1.6, 1.8) │          448 │
│ gigachat_max_26.20_uncen                 │  82.7 │ (-1.8, 1.6) │          514 │
│ vikhr-nemo-12b-instruct-r-21-09-24       │  79.8 │ (-1.6, 1.9) │          627 │
│ gemma-2-9b-it-sppo-iter3                 │  73.6 │ (-1.8, 1.2) │          509 │
│ qwen2.5-14b-instruct                     │  70.5 │ (-1.9, 1.8) │          434 │
│ gigachat_pro_26.20_uncen                 │  70.4 │ (-1.5, 2.4) │          549 │
│ gemma-2-9b-it                            │  69.2 │ (-2.0, 2.0) │          459 │
│ CohereForAI/aya-expanse-8b               │  67.1 │ (-1.6, 2.2) │          698 │
│ t-lite-instruct-0.1                      │  64.7 │ (-2.5, 2.0) │          810 │
│ vikhr-llama3.1-8b-instruct-r-21-09-24    │  63.4 │ (-2.4, 2.3) │          618 │
│ suzume-llama-3-8B-multilingual-orpo-bor… │  57.1 │ (-2.4, 2.0) │          682 │
│ gigachat_lite_26.20_uncen                │  56.4 │ (-2.2, 1.9) │          561 │
│ phi-3-medium-4k-instruct                 │  55.1 │ (-2.0, 1.9) │          566 │
│ mistral-nemo-instruct-2407               │  50.5 │ (-2.0, 2.5) │          403 │
│ yandex_gpt_pro_v4_26102024               │  50.5 │ (-2.2, 2.4) │          384 │
│ sfr-iterative-dpo-llama-3-8b-r           │  50.1 │ (-3.0, 2.3) │          516 │
│ gpt-3.5-turbo-0125                       │  50.0 │  (0.0, 0.0) │          220 │
│ glm-4-9b-chat                            │  49.8 │ (-2.5, 2.2) │          568 │
│ c4ai-command-r-v01                       │  49.0 │ (-2.3, 2.1) │          529 │
│ llama-3-instruct-8b-sppo-iter3           │  47.5 │ (-2.2, 2.2) │          502 │
│ suzume-llama-3-8b-multilingual           │  45.7 │ (-2.2, 2.0) │          641 │
│ yandex_gpt_pro                           │  45.1 │ (-2.2, 2.8) │          345 │
│ hermes-2-theta-llama-3-8b                │  44.1 │ (-2.1, 2.9) │          485 │
│ yandex_gpt_lite_v4_26102024              │  42.7 │ (-2.1, 2.4) │          328 │
│ gpt-3.5-turbo-1106                       │  41.5 │ (-2.2, 2.0) │          191 │
│ llama-3-smaug-8b                         │  40.8 │ (-2.0, 2.0) │          524 │
│ llama-3-8b-saiga-suzume-ties             │  39.9 │ (-2.1, 2.0) │          763 │
│ starling-lm-7b-beta                      │  39.8 │ (-1.9, 2.2) │          629 │
│ vikhr-it-5.4-fp16-orpo-v2                │  39.3 │ (-2.2, 2.0) │          379 │
│ saiga_llama3_8b_v6                       │  39.2 │ (-1.7, 2.1) │          471 │
│ llama-3-instruct-8b-simpo                │  38.0 │ (-1.9, 2.4) │          417 │
│ qwen2-7b-instruct                        │  37.5 │ (-2.0, 2.2) │          340 │
│ paralex-llama-3-8b-sft                   │  37.4 │ (-2.0, 1.6) │          688 │
│ aya-23-8b                                │  36.3 │ (-1.8, 1.8) │          554 │
│ meta-llama-3-8b-instruct                 │  35.1 │ (-2.2, 2.1) │          450 │
│ openchat-3.5-0106                        │  33.8 │ (-1.8, 2.1) │          492 │
│ mistral-7b-instruct-v0.3                 │  32.9 │ (-2.0, 2.0) │          469 │
│ vikhr-it-5.2-fp16-cp                     │  31.7 │ (-1.8, 1.9) │          543 │
│ hermes-2-pro-llama-3-8b                  │  30.8 │ (-1.8, 2.3) │          463 │
│ openchat-3.6-8b-20240522                 │  30.3 │ (-2.0, 2.4) │          428 │
│ vikhr-it-5.3-fp16-32k                    │  27.8 │ (-1.9, 2.1) │          519 │
│ vikhr-it-5.3-fp16                        │  22.7 │ (-1.2, 1.8) │          523 │
│ kolibri-vikhr-mistral-0427               │  22.4 │ (-1.4, 2.2) │          489 │
│ snorkel-mistral-pairrm-dpo               │  22.4 │ (-1.9, 1.6) │          773 │
│ storm-7b                                 │  20.6 │ (-1.7, 1.8) │          419 │
│ neural-chat-7b-v3-3                      │  19.0 │ (-1.7, 1.8) │          927 │└──────────────────────────────────────────┴───────┴─────────────┴──────────────┘
```

(!) - по [сообщениям](https://t.me/senior_augur/307) Ильи Гусева, модель в SFT части содержала часть ответов которые есть в бенчммарке, что могло немного завысить результаты. Подробнее в комментариях к посту.

### Со штрафом на длину ответа относительно бейзлайна
Эта функция реализована примерно как в [AlpacaEval 2 LC](https://arxiv.org/abs/2404.04475), но с некоторыми отличиями, которые можно увидеть в коде (например logistic()*2 вместо tanh(), для менее агресивного штрафования) \
Штраф применяется только к ответам где модель превосходит бейзлайн!

<details>
    <summary>Развернуть</summary>

    > python show_result.py --length-control
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
    ┃ Model                                    ┃ Score ┃      95% CI ┃ Avg. #Tokens ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
    │ gpt-4-1106-preview                       │  81.4 │ (-2.3, 2.2) │          541 │
    │ gpt-4o-mini                              │  75.4 │ (-2.0, 2.4) │          448 │
    │ gigachat_max_26.20_uncen                 │  72.5 │ (-2.5, 2.5) │          514 │
    │ vikhr-nemo-12b-instruct-r-21-09-24       │  65.5 │ (-2.1, 2.9) │          627 │
    │ qwen2.5-14b-instruct                     │  59.0 │ (-2.1, 2.0) │          434 │
    │ gemma-2-9b-it-sppo-iter3                 │  56.9 │ (-2.7, 2.1) │          509 │
    │ gigachat_pro_26.20_uncen                 │  55.4 │ (-1.8, 2.8) │          549 │
    │ gemma-2-9b-it                            │  54.3 │ (-2.0, 2.4) │          459 │
    │ gpt-3.5-turbo-0125                       │  50.0 │  (0.0, 0.0) │          220 │
    │ CohereForAI/aya-expanse-8b               │  47.2 │ (-2.1, 2.1) │          698 │
    │ vikhr-llama3.1-8b-instruct-r-21-09-24    │  46.9 │ (-2.7, 2.5) │          618 │
    │ phi-3-medium-4k-instruct                 │  45.0 │ (-2.1, 2.0) │          566 │
    │ gigachat_lite_26.20_uncen                │  41.4 │ (-2.6, 1.9) │          561 │
    │ gpt-3.5-turbo-1106                       │  41.0 │ (-2.3, 2.0) │          191 │
    │ mistral-nemo-instruct-2407               │  40.0 │ (-1.8, 2.5) │          403 │
    │ suzume-llama-3-8b-multilingual           │  40.0 │ (-2.0, 2.1) │          641 │
    │ t-lite-instruct-0.1                      │  39.9 │ (-2.5, 2.2) │          810 │
    │ yandex_gpt_pro_v4_26102024               │  38.8 │ (-1.9, 2.4) │          384 │
    │ vikhr-it-5.4-fp16-orpo-v2                │  36.8 │ (-2.0, 1.9) │          379 │
    │ glm-4-9b-chat                            │  36.2 │ (-2.3, 2.1) │          568 │
    │ yandex_gpt_pro                           │  35.3 │ (-2.0, 2.5) │          345 │
    │ hermes-2-theta-llama-3-8b                │  34.1 │ (-1.7, 2.8) │          485 │
    │ suzume-llama-3-8B-multilingual-orpo-bor… │  33.3 │ (-2.0, 2.3) │          682 │
    │ yandex_gpt_lite_v4_26102024              │  33.2 │ (-2.1, 2.3) │          328 │
    │ llama-3-smaug-8b                         │  32.5 │ (-1.8, 2.0) │          524 │
    │ sfr-iterative-dpo-llama-3-8b-r           │  32.4 │ (-2.5, 2.0) │          516 │
    │ llama-3-8b-saiga-suzume-ties             │  32.2 │ (-2.0, 1.9) │          763 │
    │ c4ai-command-r-v01                       │  32.1 │ (-2.1, 2.3) │          529 │
    │ qwen2-7b-instruct                        │  31.0 │ (-1.5, 1.7) │          340 │
    │ llama-3-instruct-8b-sppo-iter3           │  30.7 │ (-1.7, 2.1) │          502 │
    │ saiga_llama3_8b_v6                       │  30.4 │ (-1.7, 1.7) │          471 │
    │ openchat-3.5-0106                        │  30.2 │ (-1.7, 1.9) │          492 │
    │ starling-lm-7b-beta                      │  28.4 │ (-1.5, 1.8) │          629 │
    │ paralex-llama-3-8b-sft                   │  27.8 │ (-1.8, 1.5) │          688 │
    │ mistral-7b-instruct-v0.3                 │  27.8 │ (-1.8, 1.6) │          469 │
    │ hermes-2-pro-llama-3-8b                  │  26.1 │ (-1.7, 2.1) │          463 │
    │ llama-3-instruct-8b-simpo                │  25.2 │ (-1.6, 2.0) │          417 │
    │ openchat-3.6-8b-20240522                 │  24.6 │ (-1.9, 2.1) │          428 │
    │ meta-llama-3-8b-instruct                 │  23.8 │ (-1.9, 2.0) │          450 │
    │ aya-23-8b                                │  23.6 │ (-1.5, 1.5) │          554 │
    │ vikhr-it-5.2-fp16-cp                     │  23.0 │ (-1.5, 1.6) │          543 │
    │ vikhr-it-5.3-fp16-32k                    │  21.3 │ (-1.5, 1.6) │          519 │
    │ snorkel-mistral-pairrm-dpo               │  19.0 │ (-1.8, 1.3) │          773 │
    │ vikhr-it-5.3-fp16                        │  18.2 │ (-1.1, 1.4) │          523 │
    │ kolibri-vikhr-mistral-0427               │  17.8 │ (-1.4, 1.6) │          489 │
    │ neural-chat-7b-v3-3                      │  16.8 │ (-1.5, 1.5) │          927 │
    │ storm-7b                                 │  12.8 │ (-1.2, 1.3) │          419 │
    └──────────────────────────────────────────┴───────┴─────────────┴──────────────┘
</details>

Запуск `show_results.py` сохранит сгенерированные "схватки" в `data/arena_hard_battles.jsonl` и статистику бутстрапов в `data/bootstrapping_results.jsonl`. Если вы не хотите их повторно генерировать, просто переключите аргумент `--load-battles` или `--load-bootstrap` соответственно.

## Оценка собственной модели на этом бенчмарке

### Шаг 0. Установка зависимостей
```
git clone https://github.com/lm-sys/arena-hard.git
cd arena-hard
pip install -r requirements.txt
pip install -r requirements-optional.txt  # Optional dependencies (e.g., anthropic sdk)
```

### Шаг 1. Конфигурация эндпоинтов модели

Fill in your API endpoint in `config/api_config.yaml`. We support OpenAI compatible API server. You can specify `parallel` to indicate the number of concurrent API requests (default: 1).
```yaml
# example
gpt-3.5-turbo-0125:
    model_name: gpt-3.5-turbo-0125
    endpoints: null
    api_type: openai
    parallel: 5

[YOUR-MODEL-NAME]:
    model_name: [YOUR-MODEL-NAME]
    endpoints:
        - api_base: [YOUR-ENDPOINT-URL]
          api_key: [YOUR-API-KEY]
    api_type: openai
    parallel: 5
```
You may use inference engine such as [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) or [SGLang](https://github.com/sgl-project/sglang?tab=readme-ov-file#using-local-models) to host your model with an OpenAI compatible API server.


### Шаг 2. Генерация ответов модели

In `config/gen_answer_config.yaml`, add your model name in `model_list`.
```yaml
bench_name: arena-hard-v0.1
temperature: 0.0
max_tokens: 4096
num_choices: 1

model_list:
  - [YOUR-MODEL-NAME]
```
Run the command to generate answers:
```console
python gen_answer.py
```
Caching feature is implemented. The code will skip generating an answer when there is already an existing answer/judgment to the same prompt. 

### Шаг 3. Генерация судейских вердиктов

In `config/judge_config.yaml`, add your model name in `model_list`.
```yaml
...
# Add your model below for evaluation
model_list:
  - gpt-3.5-turbo-0125
  - [YOUR-MODEL-NAME]
```

Run the command to generate judgments:
```console
python gen_judgment.py
```
Judgment caching is also implemented. It will skip generating judgments that has already been generated or lacks one of the model answers.  

### Шаг 4. Отображение результата
Output model win rates.  Optionally, use `--full-stats` for detailed results.
```console
> python show_result.py
```
### Шаг 5. Arena Hard UI (экспериментально)
You can review individual judgment results using our UI code.
```console
> python qa_broswer.py --share
```
