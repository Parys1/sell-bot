import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands
from openai import OpenAI

from item_catalog import ITEM_PRICES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger('sell-bot')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
TICKETS_DB = DATA_DIR / 'tickets.json'

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN', '')
GUILD_ID = os.getenv('GUILD_ID', '')
STAFF_ROLE_ID = os.getenv('STAFF_ROLE_ID', '')
TICKETS_CATEGORY_ID = os.getenv('TICKETS_CATEGORY_ID', '')
TICKET_CHANNEL_PREFIX = os.getenv('TICKET_CHANNEL_PREFIX', 'sell')
SELL_COMMAND_CHANNEL_ID = os.getenv('SELL_COMMAND_CHANNEL_ID', '')
COMMAND_NAME = os.getenv('COMMAND_NAME', 'sell')
COMMAND_DESCRIPTION = os.getenv('COMMAND_DESCRIPTION', 'Создать заявку на продажу ресурсов')
CURRENCY_NAME = os.getenv('CURRENCY_NAME', '₽')
MAX_ACTIVE_TICKETS_PER_USER = int(os.getenv('MAX_ACTIVE_TICKETS_PER_USER', '1'))
SYNC_COMMANDS_ON_START = os.getenv('SYNC_COMMANDS_ON_START', 'true').lower() == 'true'
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')
GROQ_MAX_RETRIES = int(os.getenv('GROQ_MAX_RETRIES', '4'))
GROQ_RETRY_BASE_DELAY = float(os.getenv('GROQ_RETRY_BASE_DELAY', '1.5'))
GROQ_RETRY_MAX_DELAY = float(os.getenv('GROQ_RETRY_MAX_DELAY', '15.0'))

if not DISCORD_TOKEN:
    raise RuntimeError('Переменная окружения DISCORD_TOKEN не задана')

if not GROQ_API_KEY:
    raise RuntimeError('Переменная окружения GROQ_API_KEY не задана')


def _to_int(value: str) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


GUILD_ID_INT = _to_int(GUILD_ID)
STAFF_ROLE_ID_INT = _to_int(STAFF_ROLE_ID)
TICKETS_CATEGORY_ID_INT = _to_int(TICKETS_CATEGORY_ID)
SELL_COMMAND_CHANNEL_ID_INT = _to_int(SELL_COMMAND_CHANNEL_ID)


@dataclass
class ResourceLine:
    name: str
    amount: int
    price_per_unit: int

    @property
    def subtotal(self) -> int:
        return self.amount * self.price_per_unit


class TicketStore:
    def __init__(self, path: Path):
        self.path = path
        if not self.path.exists():
            self.path.write_text('[]', encoding='utf-8')

    def load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding='utf-8'))
        except Exception:
            logger.exception('Не удалось прочитать tickets.json, возвращаю пустой список')
            return []

    def save(self, records: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

    def append(self, record: dict[str, Any]) -> None:
        records = self.load()
        records.append(record)
        self.save(records)

    def count_active_for_user(self, user_id: int) -> int:
        records = self.load()
        return sum(1 for r in records if r.get('user_id') == user_id and r.get('status') == 'open')

    def close_by_channel(self, channel_id: int) -> bool:
        records = self.load()
        updated = False
        for record in records:
            if record.get('channel_id') == channel_id and record.get('status') == 'open':
                record['status'] = 'closed'
                record['closed_at'] = datetime.now(timezone.utc).isoformat()
                updated = True
        if updated:
            self.save(records)
        return updated


store = TicketStore(TICKETS_DB)


def slugify(text: str) -> str:
    value = text.lower().strip()
    value = re.sub(r'[^a-zA-Zа-яА-Я0-9_-]+', '-', value)
    value = re.sub(r'-{2,}', '-', value).strip('-')
    return value[:40] or 'user'


class GroqRecognizer:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_retries: int = 4,
        retry_base_delay: float = 1.5,
        retry_max_delay: float = 15.0,
    ):
        self.client = OpenAI(api_key=api_key, base_url='https://api.groq.com/openai/v1')
        self.model_name = model_name
        self.max_retries = max(1, max_retries)
        self.retry_base_delay = max(0.5, retry_base_delay)
        self.retry_max_delay = max(self.retry_base_delay, retry_max_delay)
        self.allowed_names = sorted(ITEM_PRICES.keys())
        self.response_schema = {
            'type': 'object',
            'properties': {
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'name': {'type': 'string', 'enum': self.allowed_names},
                            'amount': {'type': 'integer', 'minimum': 1},
                        },
                        'required': ['name', 'amount'],
                        'additionalProperties': False,
                    },
                },
                'analysis_note': {'type': 'string'},
            },
            'required': ['items', 'analysis_note'],
            'additionalProperties': False,
        }
        allowed = '\n'.join(f'- {name}' for name in self.allowed_names)
        self.system_prompt = (
            'Ты анализируешь скриншоты игрового инвентаря и возвращаешь только JSON по схеме. '
            'На изображении может быть один предмет или сетка из многих предметов. '
            'Нужно определить каждый видимый предмет и его количество. '
            'Количество обычно показано в правом верхнем углу ячейки в формате 1x, 23x, 46x. '
            'Если счетчик неразборчив, ставь 1. '
            'Используй только точные названия из списка ниже. '
            'Не придумывай новые названия, не добавляй цены, не пиши текст вне JSON. '
            'Если в нескольких ячейках один и тот же предмет, верни несколько элементов, код потом их суммирует. '
            'Если не уверен в каком-то предмете, пропусти его и кратко напиши об этом в analysis_note. '
            'Разрешенные названия:\n'
            f'{allowed}'
        )

    def _is_retryable_exception(self, exc: Exception) -> bool:
        message = str(exc).lower()
        markers = [
            '429',
            '500',
            '502',
            '503',
            '504',
            'rate limit',
            'service unavailable',
            'overloaded',
            'temporarily unavailable',
            'timeout',
            'timed out',
            'connection error',
            'internal server error',
            'try again',
        ]
        return any(marker in message for marker in markers)

    def _normalize_items(self, payload: dict[str, Any]) -> tuple[list[ResourceLine], str]:
        totals: dict[str, int] = defaultdict(int)
        invalid_items: list[str] = []

        for item in payload.get('items', []):
            if not isinstance(item, dict):
                continue
            name = str(item.get('name', '')).strip()
            amount_raw = item.get('amount', 1)
            if name not in ITEM_PRICES:
                if name:
                    invalid_items.append(name)
                continue
            try:
                amount = int(amount_raw)
            except (TypeError, ValueError):
                amount = 1
            if amount < 1:
                amount = 1
            totals[name] += amount

        lines = [
            ResourceLine(name=name, amount=amount, price_per_unit=ITEM_PRICES[name])
            for name, amount in sorted(totals.items())
        ]

        note = str(payload.get('analysis_note', '')).strip()
        if invalid_items:
            suffix = ' Игнорированы недопустимые названия: ' + ', '.join(sorted(set(invalid_items)))
            note = f'{note}{suffix}'.strip()
        if not note:
            note = 'Groq вернул ответ без пояснения.'
        return lines, note

    def _request_payload(self, image_url: str) -> dict[str, Any]:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': (
                                'Определи все предметы на скриншоте и их количества. '
                                'Верни только JSON по схеме. '
                                'Если на скрине много предметов, обработай всю сетку полностью.'
                            ),
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': image_url},
                        },
                    ],
                },
            ],
            temperature=0.1,
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'inventory_items',
                    'strict': False,
                    'schema': self.response_schema,
                },
            },
        )

        text = (completion.choices[0].message.content or '').strip()
        if not text:
            raise RuntimeError('Groq вернул пустой ответ')

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'Groq вернул некорректный JSON: {exc}. Ответ: {text[:500]}') from exc

    def _analyze_sync(self, image_url: str) -> tuple[list[ResourceLine], int, str]:
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                payload = self._request_payload(image_url)
                lines, note = self._normalize_items(payload)
                if not lines:
                    raise RuntimeError(f'Groq не нашел предметы. Ответ: {note}')

                total = sum(line.subtotal for line in lines)
                retry_note = '' if attempt == 1 else f' После {attempt} попыток.'
                note = f'{note} Модель: {self.model_name}. Найдено строк: {len(lines)}.{retry_note}'
                return lines, total, note
            except Exception as exc:
                last_exc = exc
                logger.warning('Groq attempt %s/%s failed: %s', attempt, self.max_retries, exc)
                if attempt >= self.max_retries or not self._is_retryable_exception(exc):
                    break
                delay = min(self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay)
                logger.info('Retrying Groq request in %.1f seconds', delay)
                time.sleep(delay)

        assert last_exc is not None
        raise RuntimeError(f'Groq недоступен или не смог корректно распознать изображение. Последняя ошибка: {last_exc}') from last_exc

    async def analyze(self, attachment: discord.Attachment) -> tuple[list[ResourceLine], int, str]:
        return await asyncio.to_thread(self._analyze_sync, attachment.url)


recognizer = GroqRecognizer(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    max_retries=GROQ_MAX_RETRIES,
    retry_base_delay=GROQ_RETRY_BASE_DELAY,
    retry_max_delay=GROQ_RETRY_MAX_DELAY,
)


class CloseTicketView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label='Закрыть тикет', style=discord.ButtonStyle.danger, emoji='🔒', custom_id='ticket:close')
    async def close_ticket(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not interaction.channel:
            return

        member = interaction.user if isinstance(interaction.user, discord.Member) else None
        has_staff = False
        if member and STAFF_ROLE_ID_INT:
            has_staff = any(role.id == STAFF_ROLE_ID_INT for role in member.roles)

        if not has_staff and not interaction.user.guild_permissions.manage_channels:
            await interaction.response.send_message('У тебя нет прав закрывать этот тикет.', ephemeral=True)
            return

        await interaction.response.send_message('Тикет будет удален через 5 секунд.', ephemeral=True)
        store.close_by_channel(interaction.channel.id)
        await asyncio.sleep(5)
        await interaction.channel.delete(reason=f'Тикет закрыт {interaction.user}')


@app_commands.command(name='ping', description='Проверить, что бот работает')
async def ping_command(interaction: discord.Interaction):
    await interaction.response.send_message('Pong! Бот работает.', ephemeral=True)


class SellBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.members = True
        super().__init__(command_prefix='!', intents=intents)
        self.tree: app_commands.CommandTree

    async def setup_hook(self) -> None:
        self.add_view(CloseTicketView())

        if GUILD_ID_INT:
            guild_obj = discord.Object(id=GUILD_ID_INT)
            self.tree.add_command(build_sell_command(), guild=guild_obj)
            self.tree.add_command(ping_command, guild=guild_obj)

            if SYNC_COMMANDS_ON_START:
                synced = await self.tree.sync(guild=guild_obj)
                logger.info('Синхронизировано %s команд в guild %s', len(synced), GUILD_ID_INT)
        else:
            self.tree.add_command(build_sell_command())
            self.tree.add_command(ping_command)

            if SYNC_COMMANDS_ON_START:
                synced = await self.tree.sync()
                logger.info('Глобально синхронизировано %s команд', len(synced))

    async def on_ready(self):
        logger.info('Бот запущен как %s (%s)', self.user, self.user.id if self.user else 'unknown')


bot = SellBot()


def build_price_block(lines: list[ResourceLine], total: int, note: str) -> str:
    if not lines:
        return f'```yaml\nРасчет: предмет не распознан\nИтог: {total} {CURRENCY_NAME}\nПримечание: {note}\n```'

    parts = [f'{line.name}: {line.amount} x {line.price_per_unit} = {line.subtotal}' for line in lines]
    joined = '\n'.join(parts)
    return f'```yaml\n{joined}\nИтог: {total} {CURRENCY_NAME}\nПримечание: {note}\n```'


async def create_ticket_channel(guild: discord.Guild, user: discord.Member, nick: str) -> discord.TextChannel:
    category = guild.get_channel(TICKETS_CATEGORY_ID_INT) if TICKETS_CATEGORY_ID_INT else None
    if category is not None and not isinstance(category, discord.CategoryChannel):
        raise RuntimeError('TICKETS_CATEGORY_ID указывает не на категорию')

    staff_role = guild.get_role(STAFF_ROLE_ID_INT) if STAFF_ROLE_ID_INT else None

    overwrites = {
        guild.default_role: discord.PermissionOverwrite(view_channel=False),
        user: discord.PermissionOverwrite(view_channel=True, send_messages=True, attach_files=True, read_message_history=True),
        guild.me: discord.PermissionOverwrite(view_channel=True, send_messages=True, manage_channels=True, attach_files=True, read_message_history=True),
    }

    if staff_role:
        overwrites[staff_role] = discord.PermissionOverwrite(view_channel=True, send_messages=True, read_message_history=True)

    channel_name = f'{TICKET_CHANNEL_PREFIX}-{slugify(nick)}-{user.id}'
    if len(channel_name) > 95:
        channel_name = channel_name[:95]

    return await guild.create_text_channel(
        name=channel_name,
        category=category,
        overwrites=overwrites,
        topic=f'Sell ticket | user={user.id} | nick={nick}'
    )


def build_sell_command() -> app_commands.Command:
    @app_commands.command(name=COMMAND_NAME, description=COMMAND_DESCRIPTION)
    @app_commands.describe(nick='Ваш игровой ник', screenshot='Прикрепите скриншот с предметами')
    async def sell_resources(interaction: discord.Interaction, nick: str, screenshot: discord.Attachment):
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message('Эту команду можно использовать только на сервере.', ephemeral=True)
            return

        if SELL_COMMAND_CHANNEL_ID_INT and interaction.channel_id != SELL_COMMAND_CHANNEL_ID_INT:
            await interaction.response.send_message(
                f'Эту команду можно использовать только в канале <#{SELL_COMMAND_CHANNEL_ID_INT}>.',
                ephemeral=True
            )
            return

        if MAX_ACTIVE_TICKETS_PER_USER > 0:
            active_count = store.count_active_for_user(interaction.user.id)
            if active_count >= MAX_ACTIVE_TICKETS_PER_USER:
                await interaction.response.send_message(
                    f'У тебя уже есть открытая заявка. Максимум активных тикетов: {MAX_ACTIVE_TICKETS_PER_USER}.',
                    ephemeral=True,
                )
                return

        if not screenshot.content_type or not screenshot.content_type.startswith('image/'):
            await interaction.response.send_message('Нужно прикрепить именно изображение.', ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True, thinking=True)

        try:
            ticket_channel = await create_ticket_channel(interaction.guild, interaction.user, nick)
        except Exception as e:
            logger.exception('Ошибка при создании тикета')
            await interaction.followup.send(f'Не удалось создать тикет: {e}', ephemeral=True)
            return

        try:
            lines, total, note = await recognizer.analyze(screenshot)
        except Exception as e:
            logger.exception('Ошибка распознавания скрина')
            lines = []
            total = 0
            note = f'Не удалось распознать предметы: {e}'

        embed = discord.Embed(
            title='Новая заявка на продажу ресурсов',
            color=discord.Color.blurple(),
            timestamp=datetime.now(timezone.utc),
        )
        embed.add_field(name='Пользователь', value=f'{interaction.user.mention} (`{interaction.user.id}`)', inline=False)
        embed.add_field(name='Игровой ник', value=discord.utils.escape_markdown(nick), inline=False)
        embed.add_field(name='Скриншот', value=f'[{screenshot.filename}]({screenshot.url})', inline=False)
        embed.add_field(name='Статус', value='Открыт', inline=True)
        embed.set_image(url=screenshot.url)
        embed.set_footer(text='Распознавание через Groq Vision')

        content_parts = []
        if STAFF_ROLE_ID_INT:
            role = interaction.guild.get_role(STAFF_ROLE_ID_INT)
            if role:
                content_parts.append(role.mention)
        content_parts.append(interaction.user.mention)
        header = ' '.join(content_parts)

        price_text = build_price_block(lines, total, note)
        view = CloseTicketView()

        message = await ticket_channel.send(content=header, embed=embed, view=view)
        await ticket_channel.send(price_text)

        record = {
            'ticket_message_id': message.id,
            'channel_id': ticket_channel.id,
            'user_id': interaction.user.id,
            'username': str(interaction.user),
            'nick': nick,
            'screenshot_url': screenshot.url,
            'resources': [asdict(line) | {'subtotal': line.subtotal} for line in lines],
            'total': total,
            'currency': CURRENCY_NAME,
            'status': 'open',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'recognizer': 'groq',
            'groq_model': GROQ_MODEL,
        }
        store.append(record)

        await interaction.followup.send(f'Готово. Тикет создан: {ticket_channel.mention}', ephemeral=True)

    return sell_resources


if __name__ == '__main__':
    bot.run(DISCORD_TOKEN)
