# NGN6 Gas Bot

Русскоязычный бот и веб-дашборд по природному газу с опорой на контракт `NGN6` в `T-Bank`.

Что сделано:

- основной инструмент: `NGN6` (`NG-7.26 Природный газ`) через `T-Bank Invest API`;
- fallback по рынку: `Yahoo Finance` для `NG=F`, `DX-Y.NYB`, `BZ=F`;
- отдельный новостной модуль: несколько RSS-лент по газу обновляются каждую минуту и сразу участвуют в новом пересчёте сигнала;
- новости получили повышенный вес в модели, поэтому свежий поток сильнее двигает итоговый bias и экстренные уведомления;
- интерфейс и API показывают сам новостной поток, его bias, силу и время последнего обновления;
- серверный watcher присылает точный дневной план в `10:20 МСК`, а утренние импульсы проверяет с `09:05 МСК` на `15m`;
- внутридневный пересчет отправляется только при смене направления, сильном сигнале или значимом изменении уровней;
- каналы уведомлений оставлены как в gold и нефтяном проекте: `Telegram`, `MAX`, `ntfy`, `webhook`.

## Запуск

```bash
npm install
npm start
```

После запуска открой [http://localhost:3002](http://localhost:3002).

Watcher уведомлений:

```bash
npm run watch:signals
```

Тест `ntfy` в правильной UTF-8 кодировке:

```bash
npm run notify:test
```

Ручная отправка текущего сигнала по новой intraday-модели:

```bash
npm run notify:current
```

Paper trading без реальных заявок:

```bash
npm run paper:trade
```

Один контрольный цикл без постоянного процесса:

```bash
npm run paper:once
```

Проверка T-Bank API, домена и TLS:

```bash
npm run tbank:check
```

Docker Compose:

```bash
docker compose up -d --build
```

Локально и в docker по умолчанию используется порт `3002`, чтобы проект мог работать рядом с `gold` и `ccm`.

## Автозапуск И Диагностика

Установка автозапуска и watchdog:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install-autostart.ps1
```

Проверка состояния после перезагрузки:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\diagnose.ps1
```

Watchdog раз в минуту проверяет `/api/health`, перезапускает dashboard при отказе, поднимает `signalWatcher` и `paperTrader`, если они не запущены. Если Планировщик Windows запрещает регистрацию задачи без прав администратора, установщик создает ярлык `NGN6-Gas-Bot-Watchdog.lnk` в пользовательской папке автозагрузки.

Для режима 24/7 каждый worker пишет heartbeat:

- `data/signal-watcher-heartbeat.json`
- `data/paper-trader-heartbeat.json`

Если цикл зависает, worker завершает себя по таймауту `180000 ms`, а watchdog поднимает его заново. Если процесс живой, но heartbeat не обновлялся дольше `WATCHDOG_HEARTBEAT_MAX_AGE_MS`, watchdog принудительно убивает зависший процесс и запускает чистый экземпляр.

## API

- `GET /api/health`
- `GET /api/news`
- `GET /api/snapshot`
- `POST /api/signal`
- `GET /api/backtest`

Все публичные API принудительно используют дневной таймфрейм.

## Дневной План

Утреннее сообщение содержит только точные уровни. Если с `09:00 МСК` появился сильный импульс, watcher может заменить нейтральный daily-фильтр на гибридный `daily + intraday` план:

- инструмент / дневной план / дата / `10:20 МСК`;
- приоритет `ЛОНГ` или `ШОРТ`;
- цена сейчас;
- разрешение лонг/шорт;
- точка входа;
- стоп;
- тейк 1;
- тейк 2;
- новостной фон.

Диапазоны входа в утренний план не отправляются.

## Внутридневные Пересчеты

Газовый контракт `NGN6` слишком волатилен для полностью статичного дневного сценария, поэтому watcher дополнительно проверяет рынок внутри дня на быстром `15m` таймфрейме:

- `09:05 МСК`: первый контроль импульса после старта;
- `09:30 МСК`: подтверждение утреннего движения;
- `10:00 МСК`: проверка перед дневным планом;
- `10:30 МСК`: контроль после дневного плана;
- `15:00 МСК`: контроль после первой части дня;
- `17:45 МСК`: контроль под американскую сессию и EIA storage по четвергам;
- `20:30 МСК`: поздний контроль только если есть сильный сценарий.

Пересчет не отправляет сообщение каждый раз. Уведомление уходит только если:

- направление изменилось относительно активного плана;
- intraday-вероятность не ниже `53%`;
- включился импульсный breakout от сессии `09:00 МСК`;
- уровни входа/стопа/тейков изменились заметно, а не косметически;
- дневной лимит торговых сигналов еще не исчерпан.

По умолчанию максимум `3` торговых сигнала в день: ранний импульс, утренний план и один подтвержденный внутридневной сигнал.

Если условия резко ухудшились против активной позиции, watcher отправляет отдельное сообщение `СРОЧНО ЗАКРЫТЬ ПОЗИЦИЮ`. Оно срабатывает, когда модель теряет направление, переворачивается против плана или сильная новость идет против текущей стороны.

При достижении `Тейк 1` watcher отправляет отдельное уведомление и переносит стоп по остатку ближе к входу. По умолчанию новый стоп считается как безубыток плюс `0.25R`, но не ставится вплотную к текущей цене.

## Paper Trading

Paper-бот не отправляет реальные заявки в T-Bank. Он каждую минуту пересчитывает intraday-сигнал, забирает живой стакан `NGN6`, виртуально открывается по ask для long и по bid для short, а закрывается обратно по bid/ask. После `takeProfit1` он частично фиксирует позицию и передвигает стоп по остатку так же, как основной watcher.

Логи пишутся в JSONL без усечения:

- `data/paper/events/YYYY-MM-DD.jsonl`
- `data/paper/orderbook/YYYY-MM-DD.jsonl`
- `data/paper-trader-state.json`
- `logs/paper-trader.out.log`
- `logs/paper-trader.err.log`

Структура стакана копится в каждом цикле: лучший bid/ask, spread, imbalance верхнего уровня, imbalance глубины, стенки bid/ask, weighted bid/ask и полный ответ T-Bank.

## T-Bank API И TLS

Проект принудительно использует REST endpoint `https://invest-public-api.tbank.ru/rest`. Старый домен `tinkoff.ru` не используется как fallback и фильтруется из `TBANK_REST_URLS`, даже если его случайно вернут в `.env`.

Для защиты от перехода T-API на сертификаты Минцифры в проект добавлен bundle `certs/russiantrustedca.pem`. Windows-скрипты запуска загружают `.env` до старта Node и автоматически выставляют `NODE_EXTRA_CA_CERTS`, если bundle найден. Docker-образ также устанавливает этот bundle в системные CA.

## State

Watcher хранит состояние в `SIGNAL_STATE_FILE`, по умолчанию:

```bash
/app/data/signal-watcher-state.json
```

В `docker-compose.yml` подключен volume:

```yaml
${SIGNAL_HOST_DATA_DIR:-./data}:/app/data
```

## Важные Переменные

Шаблон лежит в [.env.example](/C:/Users/HONOR/Documents/NGN6/.env.example).

- `TBANK_API_TOKEN`
- `TBANK_GAS_INSTRUMENT_ID`
- `TBANK_GAS_SEARCH`
- `TBANK_REST_URLS=https://invest-public-api.tbank.ru/rest`
- `DASHBOARD_HOST_PORT=3002`
- `NEWS_CACHE_MS=60000`
- `NEWS_RSS_URLS`
- `SIGNAL_DELIVERY_MODE=daily-plan`
- `SIGNAL_DAILY_PLAN_TIME=10:20`
- `SIGNAL_INTRADAY_RECHECK_TIMES=09:05,09:30,10:00,10:30,15:00,17:45,20:30`
- `SIGNAL_INTRADAY_MIN_PROBABILITY=53`
- `SIGNAL_HYBRID_MORNING_MIN_PROBABILITY=53`
- `SIGNAL_MAX_TRADE_SIGNALS_PER_DAY=3`
- `SIGNAL_IMPULSE_MOVE_PCT`
- `SIGNAL_IMPULSE_MIN_TREND`
- `SIGNAL_IMPULSE_MIN_MOMENTUM`
- `SIGNAL_TP_MANAGEMENT_ENABLED`
- `SIGNAL_TP1_STOP_LOCK_R`
- `SIGNAL_HEARTBEAT_FILE`
- `SIGNAL_CYCLE_TIMEOUT_MS`
- `PAPER_TRADER_ENABLED`
- `PAPER_POSITION_CONTRACTS`
- `PAPER_ORDERBOOK_DEPTH`
- `PAPER_TP1_FRACTION`
- `PAPER_HEARTBEAT_FILE`
- `PAPER_CYCLE_TIMEOUT_MS`
- `WATCHDOG_HEARTBEAT_MAX_AGE_MS`
- `SIGNAL_EMERGENCY_NEWS_SCORE`
- `SIGNAL_EMERGENCY_PRICE_MOVE_PCT`
- `SIGNAL_CLOSE_NEWS_CONFIDENCE`
- `NTFY_TOPIC`
- `NTFY_TITLE`
- `NTFY_RETRY_ATTEMPTS`
- `NTFY_TIMEOUT_MS`

## Ограничения

- Это decision-support tool, а не автоторговля.
- Новостной bias строится rule-based по свежим заголовкам, а не полноценной NLP-моделью.
- Бэктест не умеет ретроспективно прокручивать живые новости и поэтому оценивает только рыночную часть модели.
