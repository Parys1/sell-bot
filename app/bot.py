
import asyncio
import io
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import discord
import numpy as np
import pytesseract
from discord import app_commands
from discord.ext import commands

from item_catalog import ITEM_PRICES, TEMPLATE_ALIASES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger('sell-bot')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
TICKETS_DB = DATA_DIR / 'tickets.json'
TEMPLATES_DIR = BASE_DIR / 'app' / 'item_templates'

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

if not DISCORD_TOKEN:
    raise RuntimeError('Переменная окружения DISCORD_TOKEN не задана')


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


class ItemRecognizer:
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir
        self.templates: list[dict[str, Any]] = []
        self._load_templates()

    def _canonical_name(self, template_name: str) -> str:
        return TEMPLATE_ALIASES.get(template_name, template_name)

    def _feature_from_rgba(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x0 = max(0, int(w * 0.08))
        x1 = min(w, int(w * 0.92))
        y0 = max(0, int(h * 0.10))
        y1 = min(h, int(h * 0.78))
        rgb = image[y0:y1, x0:x1, :3]

        if image.shape[2] == 4:
            alpha = image[y0:y1, x0:x1, 3:4].astype(np.float32) / 255.0
            bg = np.full_like(rgb, 32)
            rgb = (rgb.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(gray, 50, 150)

        return np.concatenate([
            gray.astype(np.float32).flatten() / 255.0,
            edges.astype(np.float32).flatten() / 255.0,
        ])

    def _load_templates(self) -> None:
        if not self.templates_dir.exists():
            raise RuntimeError(f'Папка шаблонов не найдена: {self.templates_dir}')

        for path in sorted(self.templates_dir.glob('*.png')):
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                logger.warning('Не удалось загрузить шаблон %s', path)
                continue

            rgba = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA) if raw.shape[2] == 4 else cv2.cvtColor(raw, cv2.COLOR_BGR2RGBA)
            template_name = path.stem.lower()
            canonical_name = self._canonical_name(template_name)
            price = ITEM_PRICES.get(canonical_name)

            if price is None:
                logger.warning('У шаблона %s нет цены', template_name)
                continue

            self.templates.append({
                'template_name': template_name,
                'name': canonical_name,
                'price': price,
                'feature': self._feature_from_rgba(rgba),
            })

        if not self.templates:
            raise RuntimeError('Не удалось загрузить ни одного шаблона предметов')

        logger.info('Загружено %s шаблонов предметов', len(self.templates))

    def match(self, image_bytes: bytes) -> tuple[str, int, float]:
        raw = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError('Не удалось прочитать изображение')

        rgba = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA) if raw.shape[2] == 4 else cv2.cvtColor(raw, cv2.COLOR_BGR2RGBA)
        query_feature = self._feature_from_rgba(rgba)

        best: dict[str, Any] | None = None
        best_distance = float('inf')

        for template in self.templates:
            distance = float(np.mean((query_feature - template['feature']) ** 2))
            if distance < best_distance:
                best_distance = distance
                best = template

        if best is None:
            raise RuntimeError('Не удалось сопоставить предмет')

        confidence = max(0.0, 1.0 - min(best_distance / 0.20, 1.0))
        return best['name'], best['price'], confidence

    def extract_count(self, image_bytes: bytes) -> tuple[int | None, str]:
        raw = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            return None, ''

        h, w = raw.shape[:2]
        crop = raw[0:min(h, 26), max(0, w - 56):w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

        candidates: list[str] = []
        for invert in (False, True):
            current = 255 - enlarged if invert else enlarged
            for threshold in (90, 110, 130, 150, 170):
                _, binary = cv2.threshold(current, threshold, 255, cv2.THRESH_BINARY)
                text = pytesseract.image_to_string(
                    binary,
                    config='--psm 7 -c tessedit_char_whitelist=0123456789xX'
                )
                normalized = text.strip().lower().replace('х', 'x').replace(' ', '')
                if normalized:
                    candidates.append(normalized)

        parsed_numbers: list[int] = []
        for text in candidates:
            if text in {'x', 'xx'}:
                parsed_numbers.append(1)
                continue

            match = re.search(r'(\d+)', text)
            if match:
                try:
                    parsed_numbers.append(int(match.group(1)))
                except ValueError:
                    continue

        if not parsed_numbers:
            return None, ', '.join(candidates[:5])

        counts: dict[int, int] = {}
        for number in parsed_numbers:
            counts[number] = counts.get(number, 0) + 1

        best_amount = max(counts.items(), key=lambda item: (item[1], len(str(item[0]))))[0]
        return best_amount, ', '.join(candidates[:5])

    async def analyze(self, attachment: discord.Attachment) -> tuple[list[ResourceLine], int, str]:
        image_bytes = await attachment.read()
        item_name, price, confidence = self.match(image_bytes)
        amount, raw_count = self.extract_count(image_bytes)

        if amount is None:
            note = (
                f'Предмет распознан как "{item_name}". '
                'Количество не удалось уверенно прочитать, поставлено 1. '
                f'OCR: {raw_count or "пусто"}. '
                f'Уверенность сопоставления шаблона: {confidence:.0%}.'
            )
            amount = 1
        else:
            note = (
                f'Предмет распознан как "{item_name}". '
                f'Количество: {amount}. '
                f'Уверенность сопоставления шаблона: {confidence:.0%}.'
            )

        line = ResourceLine(name=item_name, amount=amount, price_per_unit=price)
        return [line], line.subtotal, note


recognizer = ItemRecognizer(TEMPLATES_DIR)


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
    @app_commands.describe(nick='Ваш игровой ник', screenshot='Прикрепите скриншот с предметом')
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
            note = f'Не удалось распознать предмет: {e}'

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
        embed.set_footer(text='Тестовая версия: один скрин = один предмет')

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
        }
        store.append(record)

        await interaction.followup.send(f'Готово. Тикет создан: {ticket_channel.mention}', ephemeral=True)

    return sell_resources


@app_commands.command(name='ping', description='Проверить, что бот работает')
async def ping_command(interaction: discord.Interaction):
    await interaction.response.send_message('Pong! Бот работает.', ephemeral=True)


if __name__ == '__main__':
    bot.run(DISCORD_TOKEN)
