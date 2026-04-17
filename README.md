# Discord Sell Resources Bot

Готовый Discord-бот под Railway.

## Что уже умеет

- slash-команда `/sell-resources`
- поля команды выглядят как на твоем скрине:
  - `nick`
  - `screenshot`
- создается отдельный тикет-канал
- в тикет отправляется анкета:
  - кто создал заявку
  - игровой ник
  - скриншот
  - блок расчета
- есть кнопка `Закрыть тикет`
- заявки сохраняются в `data/tickets.json`

## Что пока заглушка

Сейчас распознавание ресурсов со скриншота **не подключено**, но структура уже готова.
Позже в функцию `analyze_screenshot_placeholder()` можно добавить:

- OCR
- распознавание иконок ресурсов
- подсчет количества
- расчет цены по прайсу

## Подготовка Discord

В Discord Developer Portal у бота должны быть включены:

- `bot`
- `applications.commands`

Права бота:

- View Channels
- Send Messages
- Manage Channels
- Attach Files
- Read Message History

## Настройка Railway

Добавь переменные окружения:

- `DISCORD_TOKEN` — токен бота
- `GUILD_ID` — ID сервера для быстрой регистрации slash-команд
- `STAFF_ROLE_ID` — ID роли администрации/скупщиков
- `TICKETS_CATEGORY_ID` — ID категории, где создавать тикеты

Дополнительно можно менять:

- `TICKET_CHANNEL_PREFIX`
- `COMMAND_NAME`
- `COMMAND_DESCRIPTION`
- `CURRENCY_NAME`
- `MAX_ACTIVE_TICKETS_PER_USER`
- `SYNC_COMMANDS_ON_START`

## Локальный запуск

```bash
pip install -r requirements.txt
python app/bot.py
```

## Railway деплой

1. Загрузи проект на GitHub
2. Подключи репозиторий к Railway
3. Railway сам увидит `Procfile`
4. Добавь переменные окружения
5. Запусти deploy

## Важно по интерфейсу

Discord не позволяет сделать загрузку файла внутри modal-окна как у формы, но slash-команда с attachment-полем выглядит именно как на скрине: отдельные поля `nick` и `screenshot` с загрузкой файла.
