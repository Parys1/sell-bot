import asyncio
import base64
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

import cv2
import discord
import numpy as np
import pytesseract
from discord import app_commands
from discord.ext import commands
from openai import OpenAI

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


class TemplateIndex:
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir
        self.templates: list[dict[str, Any]] = []
        self._load_templates()

    def _canonical_name(self, template_name: str) -> str:
        return TEMPLATE_ALIASES.get(template_name, template_name)

    def _icon_region(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x0 = max(0, int(w * 0.10))
        x1 = min(w, int(w * 0.88))
        y0 = max(0, int(h * 0.16))
        y1 = min(h, int(h * 0.78))
        return image[y0:y1, x0:x1]

    def _feature(self, image: np.ndarray) -> np.ndarray:
        icon = self._icon_region(image)
        if icon.size == 0:
            icon = image
        gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(gray, 40, 120)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        vec = np.concatenate([
            gray.astype(np.float32).flatten() / 255.0,
            edges.astype(np.float32).flatten() / 255.0,
            hist.astype(np.float32),
        ])
        return vec

    def _load_templates(self) -> None:
        if not self.templates_dir.exists():
            return
        for path in sorted(self.templates_dir.glob('*.png')):
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            if raw.shape[2] == 4:
                alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
                rgb = raw[:, :, :3].astype(np.float32)
                bg = np.full_like(rgb, 32)
                bgr = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            else:
                bgr = raw[:, :, :3]
            name = self._canonical_name(path.stem.lower())
            if name not in ITEM_PRICES:
                continue
            self.templates.append({'name': name, 'feature': self._feature(bgr)})
        logger.info('Загружено шаблонов для подсказок: %s', len(self.templates))

    def candidates(self, bgr_image: np.ndarray, top_k: int = 8) -> list[tuple[str, float]]:
        if not self.templates:
            return []
        feat = self._feature(bgr_image)
        scores: list[tuple[str, float]] = []
        for template in self.templates:
            dist = float(np.mean((feat - template['feature']) ** 2))
            score = max(0.0, 1.0 - min(dist / 0.25, 1.0))
            scores.append((template['name'], score))
        by_name: dict[str, float] = {}
        for name, score in scores:
            by_name[name] = max(by_name.get(name, 0.0), score)
        return sorted(by_name.items(), key=lambda x: x[1], reverse=True)[:top_k]


def encode_png_data_uri(bgr_image: np.ndarray) -> str:
    ok, encoded = cv2.imencode('.png', bgr_image)
    if not ok:
        raise RuntimeError('Не удалось закодировать crop в PNG')
    b64 = base64.b64encode(encoded.tobytes()).decode('ascii')
    return f'data:image/png;base64,{b64}'


class HybridInventoryRecognizer:
    def __init__(self):
        self.client = OpenAI(api_key=GROQ_API_KEY, base_url='https://api.groq.com/openai/v1')
        self.model_name = GROQ_MODEL
        self.max_retries = max(1, GROQ_MAX_RETRIES)
        self.retry_base_delay = max(0.5, GROQ_RETRY_BASE_DELAY)
        self.retry_max_delay = max(self.retry_base_delay, GROQ_RETRY_MAX_DELAY)
        self.template_index = TemplateIndex(TEMPLATES_DIR)

    def _group_positions(self, positions: np.ndarray, min_gap: int = 4) -> list[int]:
        if positions.size == 0:
            return []
        groups: list[list[int]] = [[int(positions[0])]]
        for pos in positions[1:]:
            p = int(pos)
            if p - groups[-1][-1] <= min_gap:
                groups[-1].append(p)
            else:
                groups.append([p])
        return [int(round(sum(g) / len(g))) for g in groups]

    def _find_axis_lines(self, gray: np.ndarray, axis: int) -> list[int]:
        dark = (gray < 52).astype(np.float32)
        projection = dark.mean(axis=axis)
        thresholds = [0.20, 0.16, 0.12, 0.08]
        for threshold in thresholds:
            positions = np.where(projection > threshold)[0]
            grouped = self._group_positions(positions)
            if len(grouped) >= 2:
                return grouped
        return []

    def _periodic_projection_lines(self, gray: np.ndarray, axis: int, min_step: int = 46, max_step: int = 62) -> list[int]:
        dark = (gray < 60).astype(np.float32)
        projection = dark.mean(axis=axis)
        limit = projection.shape[0]
        best_score = -1.0
        best_offset = 0
        best_step = 52

        for step in range(min_step, max_step + 1):
            for offset in range(step):
                sample = projection[offset:limit:step]
                if sample.size < 3:
                    continue
                score = float(sample.mean() + sample.max() * 0.35)
                if score > best_score:
                    best_score = score
                    best_offset = offset
                    best_step = step

        lines = list(range(best_offset, limit, best_step))
        trimmed: list[int] = []
        for pos in lines:
            lo = max(0, pos - 2)
            hi = min(limit, pos + 3)
            local = projection[lo:hi]
            if local.size and float(local.max()) >= 0.05:
                trimmed.append(pos)
        return trimmed

    def _select_periodic(self, lines: list[int], min_step: int = 38, max_step: int = 90) -> list[int]:
        if len(lines) < 3:
            return lines
        best: list[int] = []
        for i in range(len(lines)):
            current = [lines[i]]
            last = lines[i]
            for j in range(i + 1, len(lines)):
                step = lines[j] - last
                if min_step <= step <= max_step:
                    current.append(lines[j])
                    last = lines[j]
            if len(current) > len(best):
                best = current
        return best or lines

    def _cell_occupied(self, cell: np.ndarray) -> bool:
        h, w = cell.shape[:2]
        icon = cell[int(h * 0.18):int(h * 0.78), int(w * 0.12):int(w * 0.88)]
        if icon.size == 0:
            return False
        gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
        return float(gray.std()) > 9.0 or float(gray.mean()) > 42.0

    def _cell_border_score(self, cell: np.ndarray) -> float:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        border = np.concatenate([
            gray[:2, :].reshape(-1),
            gray[max(0, h - 2):h, :].reshape(-1),
            gray[:, :2].reshape(-1),
            gray[:, max(0, w - 2):w].reshape(-1),
        ])
        return float((border < 70).mean())

    def _build_cells_from_lines(self, bgr: np.ndarray, x_lines: list[int], y_lines: list[int]) -> list[dict[str, Any]]:
        cells: list[dict[str, Any]] = []
        idx = 0
        for yi in range(len(y_lines) - 1):
            for xi in range(len(x_lines) - 1):
                x0, x1 = x_lines[xi], x_lines[xi + 1]
                y0, y1 = y_lines[yi], y_lines[yi + 1]
                w = x1 - x0
                h = y1 - y0
                if not (38 <= w <= 90 and 38 <= h <= 95):
                    continue
                cell = bgr[max(0, y0 + 1):min(bgr.shape[0], y1 - 1), max(0, x0 + 1):min(bgr.shape[1], x1 - 1)]
                if cell.size == 0:
                    continue
                if self._cell_border_score(cell) < 0.45:
                    continue
                if not self._cell_occupied(cell):
                    continue
                idx += 1
                cells.append({'index': idx, 'x': x0, 'y': y0, 'w': w, 'h': h, 'image': cell})
        cells.sort(key=lambda c: (c['y'], c['x']))
        return cells

    def _save_debug_grid(self, bgr: np.ndarray, x_lines: list[int], y_lines: list[int], cells: list[dict[str, Any]]) -> None:
        try:
            debug = bgr.copy()
            for x in x_lines:
                cv2.line(debug, (x, 0), (x, debug.shape[0] - 1), (255, 0, 0), 1)
            for y in y_lines:
                cv2.line(debug, (0, y), (debug.shape[1] - 1, y), (0, 255, 0), 1)
            for cell in cells:
                cv2.rectangle(debug, (cell['x'], cell['y']), (cell['x'] + cell['w'], cell['y'] + cell['h']), (0, 0, 255), 1)
            cv2.imwrite(str(DATA_DIR / 'debug_grid.png'), debug)
        except Exception:
            logger.exception('Не удалось сохранить debug_grid.png')

    def _extract_cells(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        x_lines = self._select_periodic(self._find_axis_lines(gray, axis=0))
        y_lines = self._select_periodic(self._find_axis_lines(gray, axis=1))
        cells = self._build_cells_from_lines(bgr, x_lines, y_lines)

        if len(cells) < 2:
            fallback_x = self._periodic_projection_lines(gray, axis=0)
            fallback_y = self._periodic_projection_lines(gray, axis=1)
            fallback_cells = self._build_cells_from_lines(bgr, fallback_x, fallback_y)
            if len(fallback_cells) > len(cells):
                x_lines, y_lines, cells = fallback_x, fallback_y, fallback_cells

        self._save_debug_grid(bgr, x_lines, y_lines, cells)
        return cells

    def _ocr_amount(self, cell: np.ndarray) -> tuple[int | None, str]:
        h, w = cell.shape[:2]
        crop = cell[0:max(12, int(h * 0.26)), max(0, int(w * 0.56)):w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        candidates: list[str] = []
        for invert in (False, True):
            current = 255 - gray if invert else gray
            for threshold in (90, 120, 150, 180, 210):
                _, binary = cv2.threshold(current, threshold, 255, cv2.THRESH_BINARY)
                text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789xX')
                norm = text.strip().lower().replace('х', 'x').replace(' ', '')
                if norm:
                    candidates.append(norm)
        nums: list[int] = []
        for candidate in candidates:
            match = re.search(r'(\d{1,4})', candidate)
            if match:
                try:
                    nums.append(int(match.group(1)))
                except ValueError:
                    pass
        if not nums:
            return None, ', '.join(candidates[:5]) or 'пусто'
        counts: dict[int, int] = defaultdict(int)
        for num in nums:
            counts[num] += 1
        amount = max(counts.items(), key=lambda x: (x[1], len(str(x[0]))))[0]
        return amount, ', '.join(candidates[:5])

    def _is_retryable_exception(self, exc: Exception) -> bool:
        message = str(exc).lower()
        markers = ['429', '500', '502', '503', '504', 'rate limit', 'service unavailable', 'overloaded', 'temporarily unavailable', 'timeout', 'timed out', 'connection error', 'internal server error', 'try again']
        return any(marker in message for marker in markers)

    def _classify_cell(self, cell: np.ndarray, candidates: list[tuple[str, float]]) -> tuple[str | None, float, str]:
        candidate_names = [name for name, _ in candidates][:8]
        if not candidate_names:
            return None, 0.0, 'нет кандидатов'
        data_uri = encode_png_data_uri(cell)
        schema = {
            'type': 'object',
            'properties': {
                'name': {'anyOf': [{'type': 'string', 'enum': candidate_names}, {'type': 'null'}]},
                'confidence': {'type': 'number'},
                'note': {'type': 'string'},
            },
            'required': ['name', 'confidence', 'note'],
            'additionalProperties': False,
        }
        sys_prompt = (
            'Ты видишь одну игровую ячейку инвентаря. '
            'Определи название предмета только из списка кандидатов. '
            'Счетчик количества в правом верхнем углу и вес внизу игнорируй. '
            'Смотри только на сам предмет по центру ячейки. '
            'Если не уверен, верни name=null и низкую confidence. '
            'Кандидаты:\n- ' + '\n- '.join(candidate_names)
        )
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': sys_prompt},
                        {'role': 'user', 'content': [
                            {'type': 'text', 'text': 'Определи предмет в этой одной ячейке. Верни только JSON.'},
                            {'type': 'image_url', 'image_url': {'url': data_uri}},
                        ]},
                    ],
                    temperature=0.0,
                    response_format={'type': 'json_schema', 'json_schema': {'name': 'cell_item', 'strict': False, 'schema': schema}},
                )
                text = (completion.choices[0].message.content or '').strip()
                payload = json.loads(text)
                name = payload.get('name')
                confidence = float(payload.get('confidence', 0.0) or 0.0)
                note = str(payload.get('note', '')).strip()
                if name is not None and name not in candidate_names:
                    name = None
                    confidence = 0.0
                return name, confidence, note
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries or not self._is_retryable_exception(exc):
                    break
                delay = min(self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay)
                logger.warning('Groq cell classify retry %s/%s: %s', attempt, self.max_retries, exc)
                time.sleep(delay)
        if last_exc:
            logger.warning('Groq cell classify failed: %s', last_exc)
        return None, 0.0, f'ошибка vision: {last_exc}' if last_exc else 'ошибка vision'

    def _analyze_sync(self, image_bytes: bytes) -> tuple[list[ResourceLine], int, str]:
        raw = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            raise RuntimeError('Не удалось прочитать изображение')

        cells = self._extract_cells(raw)
        if not cells:
            raise RuntimeError('Не удалось выделить ячейки инвентаря на скрине')

        totals: dict[str, int] = defaultdict(int)
        skipped: list[str] = []
        recognized = 0

        for cell_info in cells:
            cell = cell_info['image']
            amount, ocr_raw = self._ocr_amount(cell)
            if amount is None:
                amount = 1
            candidates = self.template_index.candidates(cell, top_k=8)
            name, confidence, vision_note = self._classify_cell(cell, candidates)
            if not name or confidence < 0.45:
                cand_text = '; '.join(f'{n} ({s:.3f})' for n, s in candidates[:3]) or 'нет'
                skipped.append(
                    f'ячейка {cell_info["index"]}: неуверенно, OCR={amount}, top={cand_text}, vision={vision_note or "пусто"}'
                )
                continue
            totals[name] += amount
            recognized += 1

        lines = [
            ResourceLine(name=name, amount=amount, price_per_unit=ITEM_PRICES[name])
            for name, amount in sorted(totals.items())
        ]
        total = sum(line.subtotal for line in lines)

        if not lines:
            raise RuntimeError('Ни один предмет не распознан уверенно. ' + ' | '.join(skipped[:8]))

        note = f'Распознано ячеек: {recognized}/{len(cells)}'
        if skipped:
            note += f' | пропущено: {len(skipped)} | ' + ' | '.join(skipped[:6])
        return lines, total, note

    async def analyze(self, attachment: discord.Attachment) -> tuple[list[ResourceLine], int, str]:
        image_bytes = await attachment.read()
        return await asyncio.to_thread(self._analyze_sync, image_bytes)


recognizer = HybridInventoryRecognizer()


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
        return f'```yaml\nРасчет: предметы не распознаны\nИтог: {total} {CURRENCY_NAME}\nПримечание: {note}\n```'
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
    return await guild.create_text_channel(name=channel_name, category=category, overwrites=overwrites, topic=f'Sell ticket | user={user.id} | nick={nick}')


def build_sell_command() -> app_commands.Command:
    @app_commands.command(name=COMMAND_NAME, description=COMMAND_DESCRIPTION)
    @app_commands.describe(nick='Ваш игровой ник', screenshot='Прикрепите скриншот с предметами')
    async def sell_resources(interaction: discord.Interaction, nick: str, screenshot: discord.Attachment):
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message('Эту команду можно использовать только на сервере.', ephemeral=True)
            return
        if SELL_COMMAND_CHANNEL_ID_INT and interaction.channel_id != SELL_COMMAND_CHANNEL_ID_INT:
            await interaction.response.send_message(f'Эту команду можно использовать только в канале <#{SELL_COMMAND_CHANNEL_ID_INT}>.', ephemeral=True)
            return
        if MAX_ACTIVE_TICKETS_PER_USER > 0:
            active_count = store.count_active_for_user(interaction.user.id)
            if active_count >= MAX_ACTIVE_TICKETS_PER_USER:
                await interaction.response.send_message(f'У тебя уже есть открытая заявка. Максимум активных тикетов: {MAX_ACTIVE_TICKETS_PER_USER}.', ephemeral=True)
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

        embed = discord.Embed(title='Новая заявка на продажу ресурсов', color=discord.Color.blurple(), timestamp=datetime.now(timezone.utc))
        embed.add_field(name='Пользователь', value=f'{interaction.user.mention} (`{interaction.user.id}`)', inline=False)
        embed.add_field(name='Игровой ник', value=discord.utils.escape_markdown(nick), inline=False)
        embed.add_field(name='Скриншот', value=f'[{screenshot.filename}]({screenshot.url})', inline=False)
        embed.add_field(name='Статус', value='Открыт', inline=True)
        embed.set_image(url=screenshot.url)
        embed.set_footer(text='Распознавание: OCR количества + Groq по отдельным ячейкам')

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
            'recognizer': 'hybrid_groq_ocr',
            'groq_model': GROQ_MODEL,
        }
        store.append(record)
        await interaction.followup.send(f'Готово. Тикет создан: {ticket_channel.mention}', ephemeral=True)

    return sell_resources


if __name__ == '__main__':
    bot.run(DISCORD_TOKEN)
