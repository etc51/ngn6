# NGN6 Intraday Trading Bot

Рабочий прототип торгового бота для фьючерса NGN6 на Московской бирже через T-Invest API.

По умолчанию бот работает только в `dry_run` + paper-режиме: он читает рыночные данные, строит сигналы, ведет виртуальный счет и пишет структурированные данные, но не выставляет реальные заявки. Реальная торговля включается только явным `dry_run: false` в конфиге.

## Архитектура

Поток данных:

1. `TInvestGateway` открывает MarketData stream для стакана, свечей и сделок.
2. Если stream не дает свежих данных дольше `polling.stale_after_seconds`, включается polling-страховка через unary-запросы рынка.
3. `MarketState` хранит последние 1m/5m/15m свечи, стакан и сделки.
4. `indicators.py` считает EMA5/EMA10, RSI14 и Bollinger Bands.
5. `orderbook.py` оценивает спред, дисбаланс, крупные плотности и съедание плотностей.
6. `signals.py` строит long/short/flat решение по EMA, RSI, объему, 15m-контексту, волатильности и стакану.
7. `risk.py` рассчитывает размер позиции, stop-loss, partial take-profit и trailing-stop.
8. `execution.py` исполняет заявки через paper/dry-run симулятор или T-Invest adapter.
9. `paper.py` ведет виртуальный счет: капитал 300 000 RUB и лимит маржинальной экспозиции 1 500 000 RUB.
10. `recorder.py` пишет market structure и generated decisions в JSONL для анализа и последующего обучения.
11. `review.py` строит review-графики по расписанию 12:00 и 19:00.
12. `dashboard.py` показывает paper equity, позицию, структуру рынка, решения и review charts.

## Конфиг

Основной файл: `config/ngn6.yaml`.

Ключевые runtime-секции:

- `paper`: виртуальный депозит, лимит маржи и файл состояния `data/paper_state.json`;
- `data_collection`: JSONL-файлы `data/market_structure.jsonl` и `data/decisions.jsonl`;
- `review`: расписание графиков `12:00` и `19:00`, каталог `reports/review`;
- `dashboard`: host/port локальной панели.

Токен не хранится в коде. Бот читает его из `T_INVEST_TOKEN`, либо из файла `auth.token_file`.
Для вашего случая в конфиге уже указан шаблон пути к файлу на рабочем столе:

```yaml
auth:
  token_env: T_INVEST_TOKEN
  token_file: "%USERPROFILE%/Desktop/жрт новый про.txt"
```

Если имя файла отличается, поменяйте только путь. Значение токена в репозиторий не добавляйте.

## Запуск

Установка:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .[dev]
```

`requirements.txt` подключает официальный registry T-Bank для пакета `t-tech-investments`.

Dry-run:

```powershell
ngn6-bot run --config config/ngn6.yaml
```

Dashboard:

```powershell
ngn6-bot dashboard --config config/ngn6.yaml --host 127.0.0.1 --port 8080
```

Docker:

```powershell
docker build -t ngn6-bot .
docker run --rm --env-file .env ngn6-bot
```

Linux server: шаблоны systemd unit лежат в `deployment/ngn6-bot.service` и `deployment/ngn6-dashboard.service`.

Проверка конфигурации и импорта без подключения к бирже:

```powershell
ngn6-bot smoke --config config/ngn6.yaml
```

Read-only проверка токена, инструмента и стакана без заявок:

```powershell
ngn6-bot check-api --config config/ngn6.yaml
```

Read-only проверка API, загрузки истории и одного dry-run цикла стратегии:

```powershell
ngn6-bot check-strategy --config config/ngn6.yaml
```

Read-only проверка market data stream:

```powershell
ngn6-bot check-stream --config config/ngn6.yaml --seconds 15
```

Тесты:

```powershell
pytest
```

Backtest/replay по свежим 1m свечам T-Invest:

```powershell
ngn6-bot backtest --config config/ngn6.yaml --minutes 4500 --report reports/backtest.json
```

Walk-forward по хронологическим фолдам:

```powershell
ngn6-bot walk-forward --config config/ngn6.yaml --minutes 4500 --folds 3 --report reports/walk_forward.json
```

График с индикаторами за дату:

```powershell
ngn6-bot chart --config config/ngn6.yaml --date 2026-06-29 --timeframe 15min
```

Review-графики вручную:

```powershell
ngn6-bot review --config config/ngn6.yaml --date 2026-06-30 --label manual
```

При обычном `ngn6-bot run` review-графики строятся автоматически в 12:00 и 19:00 по timezone из конфига.

Ограничение: исторический стакан через текущий API-контур не реплеится, поэтому backtest проверяет свечную часть стратегии, риск и клиринг. Order book absorption проверяется live dry-run через `check-stream` и основной runtime.

## Стратегия

Цель: внутридневные движения 2-3% без переноса позиции через клиринг.

Лонг:

- 15m контекст должен быть восходящим: цена выше 15m EMA5/EMA10, EMA5 выше EMA10, EMA не разворачиваются вниз;
- цена удерживается выше EMA5/EMA10 на 1m/5m;
- объем выше среднего или в стакане есть подтверждение;
- спред ниже лимита;
- Bollinger width не ниже порога;
- ручной news-halt выключен;
- RSI не используется как единственная причина контртрендового входа.

Шорт симметричен: цена ниже EMA5/EMA10, подтверждение объемом/стаканом, сопротивление или пробой вниз.
При `signals.require_15m_direction: true` 1m/5m сигнал не откроет сделку против 15m направления.

## Риск

- риск на сделку: `risk.risk_per_trade_pct`, по умолчанию 1%;
- максимальный риск ограничен `risk.max_risk_per_trade_pct`;
- stop-loss за локальным уровнем или экстремумом сигнальной свечи;
- 50% позиции фиксируется при 1-1.5% прибыли;
- остаток сопровождается trailing stop;
- позиции закрываются до клиринга по расписанию из `session.clearings`.

## План тестирования

1. Unit tests: индикаторы, стакан, сигналы, риск, конфиг.
2. Paper/dry-run на live stream: проверить стабильность стрима, fallback polling, спреды, задержки и качество логов.
3. Исторический replay свечей и синтетического стакана: оценить правила входа/выхода без реальных заявок.
4. Метрики: средняя сделка, profit factor, win rate, max drawdown, среднее/макс. проскальзывание, доля пропущенных сделок из-за спреда/волатильности/news-halt.
5. Отдельно проверять интервалы низкой ликвидности, клиринг, утренние гэпы и расширение спреда.

## Риски и ограничения

- NG может резко двигаться на внешних новостях, а ручной news-halt не защищает от внезапных событий.
- В газе возможны расширение спреда, низкая глубина и сильное проскальзывание.
- Данные стакана показывают лимитные заявки, часть плотностей может быть быстро снята.
- T-Invest API и биржевые ограничения могут менять доступные инструменты, режимы заявок и расписание.
- Исторические свечи без реального стакана недостаточны для доказательства прибыльности.
- Реальная торговля требует отдельного контура мониторинга, алертов, лимитов по дневной просадке и kill-switch.
## Codex Folder Index

This repository now has folder-level README files for fast navigation.

- `config/` - runtime configuration and safety thresholds.
- `deployment/` - systemd service templates.
- `ngn6_bot/` - active Python bot package.
- `ngn6_bot/learning/` - ML, labels, shadow mode, promotion gates.
- `ngn6_bot/monitoring/` - drift and monitoring helpers.
- `scripts/` - local Windows operational scripts.
- `tests/` - Python test suite.
- `reference/` - legacy/reference source only.

Repository rules:

- Active collection runs on the VPS, not the local Windows machine.
- Runtime outputs live in `data/`, `logs/`, and `reports/`; they are ignored by git.
- Every runtime JSON/JSONL report should include `commit_hash`.
- Do not commit real tokens, account IDs, model artifacts, or market-data dumps.
