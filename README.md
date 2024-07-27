# Русский SBS бенчмарк на основе кода Arena-Hard-Auto

### Описание
Это инструмент для **автоматической** оценки моделей на **русском языке** с помощью **сильной LLM** **(GPT-4-1106-preview)**. Использует систему [ELO рангов](https://ru.wikipedia.org/wiki/Рейтинг_Эло).

Основывается на фиксированном наборе из 500 промптов, разбитым по 50 темам. Каждая модель дает свой ответ на каждый промпт,
после чего он сравнивается с ответами на эти же промпты от **модели-бейзлайна (gpt-3.5-turbo-0125)**.

Важными особенностями отличающими Arena-Hard-Auto от обычного SBS ялвются:
- При сравнениях ответов учитываются 3 основных случая: >> (сильно лучше), > (просто лучше) и = (примерно одинаково), за случаи когда один ответ сильно лучше другого вес вердикта увеличивается в 3 раза
- Для удаления позиционного биаса в промпте модели-судьи, каждое сравнение делается 2 раза (ответы моделей переставляются местами в промпте).
- Бутстрапирование результатов сравнений для получения доверительных интервалов
- Использование системы ELO рангов и предсказения винрейта с помощью [Bradley–Terry модели](https://en.wikipedia.org/wiki/Bradley–Terry_model)

В отличие от [оригинала Arena-Hard-Auto](https://github.com/lm-sys/arena-hard-auto), этот репозиторий содержит некоторые изменения:
1. Изменен промпт для модели-оценщика, для того чтобы сравнивать модели в том числе по владению русским языком, сам промпт находится в `config/judge_config.yaml`
2. Добавлена функция контроля длины ответов для штрафования за слишком длинные ответы по сравнению с бейзлайном (экспериментально)
3. В качестве бейзлайна используется gpt-3.5-turbo-0125, в отличие от GPT-4, так как для русского языка модели менее развиты чем для английского
4. Добавлены функции генерации с gigachat и yandexgpt
5. Фиксы некоторых багов в оригинальной имплементации

### Датасет с промптами
Для этой арены существует 2 датасета, но используется сейчас только первый:
1. [General](https://huggingface.co/datasets/Vikhrmodels/ru-arena-general) (Диверсифицированные по 50 топикам вопросы из онлайн lmsys арены) - Именно он используется сейчас
2. [Hard](https://huggingface.co/datasets/Vikhrmodels/ru-arena-hard) (Переведенный оригинальный датасет из английской Arena-Hard)

## Состояние арены на 26.07.2024

На текущий момент в рейтинге находятся 39 моделей \
`score` - предсказаный винрейт модели относительно бейзлайна

### Без контроля длины
```console
> python show_result.py
gpt-4-1106-preview                                 | score: 90.9  | 95% CI: (-1.1, 1.0)  | average #tokens: 541
gpt-4o-mini                                        | score: 83.9  | 95% CI: (-1.3, 1.4)  | average #tokens: 448
gemma-2-9b-it-sppo-iter3                           | score: 73.6  | 95% CI: (-1.8, 1.9)  | average #tokens: 509
gemma-2-9b-it                                      | score: 69.2  | 95% CI: (-2.2, 1.6)  | average #tokens: 459
t-lite-instruct-0.1                                | score: 64.7  | 95% CI: (-1.6, 2.0)  | average #tokens: 810
suzume-llama-3-8B-multilingual-orpo-borda-half     | score: 57.1  | 95% CI: (-2.5, 2.5)  | average #tokens: 682
phi-3-medium-4k-instruct                           | score: 55.1  | 95% CI: (-2.3, 2.6)  | average #tokens: 566
mistral-nemo-instruct-2407                         | score: 50.5  | 95% CI: (-1.6, 2.4)  | average #tokens: 403
sfr-iterative-dpo-llama-3-8b-r                     | score: 50.1  | 95% CI: (-2.1, 2.4)  | average #tokens: 516
gpt-3.5-turbo-0125                                 | score: 50.0  | 95% CI:  (0.0, 0.0)  | average #tokens: 220
glm-4-9b-chat                                      | score: 49.8  | 95% CI: (-2.5, 2.2)  | average #tokens: 568
c4ai-command-r-v01                                 | score: 49.0  | 95% CI: (-2.0, 2.0)  | average #tokens: 529
llama-3-instruct-8b-sppo-iter3                     | score: 47.5  | 95% CI: (-2.4, 2.1)  | average #tokens: 502
suzume-llama-3-8b-multilingual                     | score: 45.7  | 95% CI: (-2.3, 1.9)  | average #tokens: 641
yandex_gpt_pro                                     | score: 45.1  | 95% CI: (-2.3, 2.1)  | average #tokens: 345
hermes-2-theta-llama-3-8b                          | score: 44.1  | 95% CI: (-2.1, 2.2)  | average #tokens: 485
gpt-3.5-turbo-1106                                 | score: 41.5  | 95% CI: (-2.0, 2.8)  | average #tokens: 191
llama-3-smaug-8b                                   | score: 40.8  | 95% CI: (-2.5, 2.6)  | average #tokens: 524
llama-3-8b-saiga-suzume-ties                       | score: 39.9  | 95% CI: (-2.2, 2.0)  | average #tokens: 763
starling-lm-7b-beta                                | score: 39.8  | 95% CI: (-2.6, 2.1)  | average #tokens: 629
vikhr-it-5.4-fp16-orpo-v2                          | score: 39.3  | 95% CI: (-2.1, 2.3)  | average #tokens: 379
saiga_llama3_8b_v6                                 | score: 39.2  | 95% CI: (-2.2, 2.2)  | average #tokens: 471
llama-3-instruct-8b-simpo                          | score: 38.0  | 95% CI: (-2.5, 2.1)  | average #tokens: 417
qwen2-7b-instruct                                  | score: 37.5  | 95% CI: (-2.6, 2.3)  | average #tokens: 340
paralex-llama-3-8b-sft                             | score: 37.4  | 95% CI: (-2.1, 2.4)  | average #tokens: 688
aya-23-8b                                          | score: 36.3  | 95% CI: (-1.9, 2.0)  | average #tokens: 554
meta-llama-3-8b-instruct                           | score: 35.1  | 95% CI: (-2.4, 2.2)  | average #tokens: 450
openchat-3.5-0106                                  | score: 33.8  | 95% CI: (-2.1, 2.2)  | average #tokens: 492
mistral-7b-instruct-v0.3                           | score: 32.9  | 95% CI: (-1.7, 1.9)  | average #tokens: 469
vikhr-it-5.2-fp16-cp                               | score: 31.7  | 95% CI: (-2.5, 1.9)  | average #tokens: 543
gigachat_pro                                       | score: 31.4  | 95% CI: (-2.2, 1.8)  | average #tokens: 294
hermes-2-pro-llama-3-8b                            | score: 30.8  | 95% CI: (-2.0, 2.2)  | average #tokens: 463
openchat-3.6-8b-20240522                           | score: 30.3  | 95% CI: (-1.7, 2.2)  | average #tokens: 428
vikhr-it-5.3-fp16-32k                              | score: 27.8  | 95% CI: (-1.8, 1.8)  | average #tokens: 519
vikhr-it-5.3-fp16                                  | score: 22.7  | 95% CI: (-1.8, 1.6)  | average #tokens: 523
kolibri-vikhr-mistral-0427                         | score: 22.4  | 95% CI: (-1.8, 2.1)  | average #tokens: 489
snorkel-mistral-pairrm-dpo                         | score: 22.4  | 95% CI: (-1.6, 1.6)  | average #tokens: 773
storm-7b                                           | score: 20.6  | 95% CI: (-1.6, 1.6)  | average #tokens: 419
neural-chat-7b-v3-3                                | score: 19.0  | 95% CI: (-1.4, 2.1)  | average #tokens: 927
gigachat_lite                                      | score: 17.2  | 95% CI: (-1.7, 1.6)  | average #tokens: 276
```

### Со штрафом на длину ответа относительно бейзлайна
Эта функция реализована примерно как в [AlpacaEval 2 LC](https://arxiv.org/abs/2404.04475), но с некоторыми отличиями, которые можно увидеть в коде (например logistic()*2 вместо tanh(), для менее агресивного штрафования) \
Штраф применяется только к ответам где модель превосходит бейзлайн!

<details>
    <summary>Развернуть</summary>

    > python show_result.py --length-control
    gpt-4-1106-preview                                 | score: 81.4  | 95% CI: (-2.0, 1.9)  | average #tokens: 541
    gpt-4o-mini                                        | score: 75.4  | 95% CI: (-1.8, 1.6)  | average #tokens: 448
    gemma-2-9b-it-sppo-iter3                           | score: 56.9  | 95% CI: (-2.3, 2.7)  | average #tokens: 509
    gemma-2-9b-it                                      | score: 54.3  | 95% CI: (-2.3, 2.0)  | average #tokens: 459
    gpt-3.5-turbo-0125                                 | score: 50.0  | 95% CI:  (0.0, 0.0)  | average #tokens: 220
    phi-3-medium-4k-instruct                           | score: 45.0  | 95% CI: (-2.4, 2.7)  | average #tokens: 566
    gpt-3.5-turbo-1106                                 | score: 41.0  | 95% CI: (-2.2, 3.0)  | average #tokens: 191
    mistral-nemo-instruct-2407                         | score: 40.0  | 95% CI: (-1.8, 2.4)  | average #tokens: 403
    suzume-llama-3-8b-multilingual                     | score: 40.0  | 95% CI: (-2.0, 1.8)  | average #tokens: 641
    t-lite-instruct-0.1                                | score: 39.9  | 95% CI: (-2.2, 2.4)  | average #tokens: 810
    vikhr-it-5.4-fp16-orpo-v2                          | score: 36.8  | 95% CI: (-1.9, 2.1)  | average #tokens: 379
    glm-4-9b-chat                                      | score: 36.2  | 95% CI: (-2.1, 2.1)  | average #tokens: 568
    yandex_gpt_pro                                     | score: 35.3  | 95% CI: (-2.2, 1.8)  | average #tokens: 345
    hermes-2-theta-llama-3-8b                          | score: 34.1  | 95% CI: (-1.7, 2.6)  | average #tokens: 485
    suzume-llama-3-8B-multilingual-orpo-borda-half     | score: 33.3  | 95% CI: (-2.3, 2.5)  | average #tokens: 682
    llama-3-smaug-8b                                   | score: 32.5  | 95% CI: (-2.3, 2.2)  | average #tokens: 524
    sfr-iterative-dpo-llama-3-8b-r                     | score: 32.4  | 95% CI: (-2.1, 1.9)  | average #tokens: 516
    llama-3-8b-saiga-suzume-ties                       | score: 32.2  | 95% CI: (-1.9, 2.0)  | average #tokens: 763
    c4ai-command-r-v01                                 | score: 32.1  | 95% CI: (-1.9, 2.2)  | average #tokens: 529
    qwen2-7b-instruct                                  | score: 31.0  | 95% CI: (-2.3, 1.9)  | average #tokens: 340
    llama-3-instruct-8b-sppo-iter3                     | score: 30.7  | 95% CI: (-2.1, 2.0)  | average #tokens: 502
    saiga_llama3_8b_v6                                 | score: 30.4  | 95% CI: (-1.8, 2.1)  | average #tokens: 471
    openchat-3.5-0106                                  | score: 30.2  | 95% CI: (-1.8, 1.9)  | average #tokens: 492
    starling-lm-7b-beta                                | score: 28.4  | 95% CI: (-2.1, 1.8)  | average #tokens: 629
    paralex-llama-3-8b-sft                             | score: 27.8  | 95% CI: (-2.0, 1.9)  | average #tokens: 688
    mistral-7b-instruct-v0.3                           | score: 27.8  | 95% CI: (-1.7, 1.8)  | average #tokens: 469
    hermes-2-pro-llama-3-8b                            | score: 26.1  | 95% CI: (-1.8, 1.9)  | average #tokens: 463
    llama-3-instruct-8b-simpo                          | score: 25.2  | 95% CI: (-2.1, 1.6)  | average #tokens: 417
    gigachat_pro                                       | score: 24.7  | 95% CI: (-1.9, 1.4)  | average #tokens: 294
    openchat-3.6-8b-20240522                           | score: 24.6  | 95% CI: (-1.5, 2.0)  | average #tokens: 428
    meta-llama-3-8b-instruct                           | score: 23.8  | 95% CI: (-2.0, 1.6)  | average #tokens: 450
    aya-23-8b                                          | score: 23.6  | 95% CI: (-1.6, 1.4)  | average #tokens: 554
    vikhr-it-5.2-fp16-cp                               | score: 23.0  | 95% CI: (-1.9, 1.9)  | average #tokens: 543
    vikhr-it-5.3-fp16-32k                              | score: 21.3  | 95% CI: (-1.8, 1.6)  | average #tokens: 519
    snorkel-mistral-pairrm-dpo                         | score: 19.0  | 95% CI: (-1.5, 1.5)  | average #tokens: 773
    vikhr-it-5.3-fp16                                  | score: 18.2  | 95% CI: (-1.6, 1.4)  | average #tokens: 523
    kolibri-vikhr-mistral-0427                         | score: 17.8  | 95% CI: (-1.5, 1.8)  | average #tokens: 489
    neural-chat-7b-v3-3                                | score: 16.8  | 95% CI: (-1.3, 1.8)  | average #tokens: 927
    gigachat_lite                                      | score: 15.2  | 95% CI: (-1.6, 1.6)  | average #tokens: 276
    storm-7b                                           | score: 12.8  | 95% CI: (-1.2, 1.2)  | average #tokens: 419
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
