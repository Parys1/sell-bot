import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

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
COMMAND_NAME = os.getenv('COMMAND_NAME', 'sell-resources')
COMMAND_DESCRIPTION = os.getenv('COMMAND_DESCRIPTION', 'Создать заявку на продажу ресурсов')
CURRENCY_NAME = os.getenv('CURRENCY_NAME', '₽')
MAX_ACTIVE_TICKETS_PER_USER = int(os.getenv('MAX_ACTIVE_TICKETS_PER_USER', '1'))
CLOSE_ON_START = os.getenv('SYNC_COMMANDS_ON_START', 'true').lower() == 'true'

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
        self.path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

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


async def analyze_screenshot_placeholder(attachment: discord.Attachment) -> tuple[list[ResourceLine], int, str]:
    """
    Заглушка.
    Позже здесь можно подключить OCR / CV / ML распознавание предметов со скрина.
    Сейчас бот работает полностью, но расчет возвращает демо-статус.
    """
    await asyncio.sleep(0.2)
    demo_lines: list[ResourceLine] = []
    total = sum(line.subtotal for line in demo_lines)
    note = (
        'Авторасчет пока не подключен. '\
        'Структура готова: сюда позже добавим распознавание ресурсов, количества и цен.'
    )
    return demo_lines, total, note


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
        self.tree.add_command(build_sell_command())

        if CLOSE_ON_START:
            if GUILD_ID_INT:
                guild_obj = discord.Object(id=GUILD_ID_INT)
                self.tree.copy_global_to(guild=guild_obj)
                synced = await self.tree.sync(guild=guild_obj)
                logger.info('Синхронизировано %s команд в guild %s', len(synced), GUILD_ID_INT)
            else:
                synced = await self.tree.sync()
                logger.info('Глобально синхронизировано %s команд', len(synced))

    async def on_ready(self):
        logger.info('Бот запущен как %s (%s)', self.user, self.user.id if self.user else 'unknown')


bot = SellBot()


def build_price_block(lines: list[ResourceLine], total: int, note: str) -> str:
    if not lines:
        return f'```yaml\nРасчет: ожидает модуль распознавания\nИтог: {total} {CURRENCY_NAME}\nПримечание: {note}\n```'

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

    channel = await guild.create_text_channel(
        name=channel_name,
        category=category,
        overwrites=overwrites,
        topic=f'Sell ticket | user={user.id} | nick={nick}'
    )
    return channel


def build_sell_command() -> app_commands.Command:
    @app_commands.command(name=COMMAND_NAME, description=COMMAND_DESCRIPTION)
    @app_commands.describe(
        nick='Ваш игровой ник',
        screenshot='Прикрепите скриншот с ресурсами'
    )
    async def sell_resources(
        interaction: discord.Interaction,
        nick: str,
        screenshot: discord.Attachment,
    ):
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message('Эту команду можно использовать только на сервере.', ephemeral=True)
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

        lines, total, note = await analyze_screenshot_placeholder(screenshot)

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
        embed.set_footer(text='После подключения OCR здесь будет полный авторасчет')

        content_parts = []
        if STAFF_ROLE_ID_INT:
            role = interaction.guild.get_role(STAFF_ROLE_ID_INT)
            if role:
                content_parts.append(role.mention)
        content_parts.append(interaction.user.mention)
        header = ' '.join(content_parts)

        price_text = build_price_block(lines, total, note)
        view = CloseTicketView()

        message = await ticket_channel.send(
            content=header,
            embed=embed,
            view=view,
        )
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

        await interaction.followup.send(
            f'Готово. Тикет создан: {ticket_channel.mention}',
            ephemeral=True,
        )

    return sell_resources


@bot.tree.command(name='ping', description='Проверить, что бот работает')
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message('Pong! Бот работает.', ephemeral=True)


if __name__ == '__main__':
    bot.run(DISCORD_TOKEN)
