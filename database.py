"""
Voice Coach 数据库模块
使用 SQLite 存储转录片段和分析报告
"""

import json
import sqlite3
import logging
from datetime import datetime, date
from typing import Optional

import config

logger = logging.getLogger("voice_coach.database")


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接"""
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """初始化数据库表结构"""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_path TEXT,
                started_at TEXT,
                ended_at TEXT,
                duration_s REAL,
                transcript TEXT,
                char_count INTEGER DEFAULT 0,
                info_density REAL DEFAULT 0.0,
                is_valid INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start TEXT,
                period_end TEXT,
                segment_ids TEXT,
                total_segments INTEGER DEFAULT 0,
                total_chars INTEGER DEFAULT 0,
                report TEXT,
                emailed INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            );

            CREATE INDEX IF NOT EXISTS idx_segments_started_at ON segments(started_at);
            CREATE INDEX IF NOT EXISTS idx_segments_is_valid ON segments(is_valid);
            CREATE INDEX IF NOT EXISTS idx_analyses_period ON analyses(period_start, period_end);
        """)
        conn.commit()
        logger.info("数据库初始化完成: %s", config.DB_PATH)
    finally:
        conn.close()


def save_segment(
    audio_path: str,
    started_at: str,
    ended_at: str,
    duration_s: float,
    transcript: str,
    char_count: int,
    info_density: float,
    is_valid: bool,
) -> int:
    """保存一个转录片段，返回 id"""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            """INSERT INTO segments
               (audio_path, started_at, ended_at, duration_s, transcript,
                char_count, info_density, is_valid)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                audio_path,
                started_at,
                ended_at,
                duration_s,
                transcript,
                char_count,
                round(info_density, 4),
                1 if is_valid else 0,
            ),
        )
        conn.commit()
        seg_id = cursor.lastrowid
        logger.debug("保存片段 #%d (有效=%s, 字数=%d, 密度=%.2f)",
                      seg_id, is_valid, char_count, info_density)
        return seg_id
    finally:
        conn.close()


def get_valid_segments(since: str, until: str) -> list[dict]:
    """获取指定时间范围内的有效片段"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT id, audio_path, started_at, ended_at, duration_s,
                      transcript, char_count, info_density
               FROM segments
               WHERE is_valid = 1 AND started_at >= ? AND started_at <= ?
               ORDER BY started_at""",
            (since, until),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_analysis(
    period_start: str,
    period_end: str,
    segment_ids: list[int],
    total_segments: int,
    total_chars: int,
    report: str,
) -> int:
    """保存分析报告，返回 id"""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            """INSERT INTO analyses
               (period_start, period_end, segment_ids, total_segments,
                total_chars, report)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                period_start,
                period_end,
                json.dumps(segment_ids),
                total_segments,
                total_chars,
                report,
            ),
        )
        conn.commit()
        analysis_id = cursor.lastrowid
        logger.info("保存分析报告 #%d (片段数=%d, 总字数=%d)",
                     analysis_id, total_segments, total_chars)
        return analysis_id
    finally:
        conn.close()


def mark_emailed(analysis_id: int):
    """标记分析报告已发送邮件"""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE analyses SET emailed = 1 WHERE id = ?",
            (analysis_id,),
        )
        conn.commit()
        logger.debug("分析报告 #%d 已标记为已发送", analysis_id)
    finally:
        conn.close()


def get_last_analysis_end() -> str | None:
    """获取最近一次分析的结束时间，用于增量分析"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT period_end FROM analyses ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            return row["period_end"]
        return None
    finally:
        conn.close()


def get_today_stats() -> dict:
    """获取今日统计数据"""
    today_str = date.today().isoformat()
    since = f"{today_str} 00:00:00"
    until = f"{today_str} 23:59:59"

    conn = _get_conn()
    try:
        # 总片段数
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM segments WHERE started_at >= ? AND started_at <= ?",
            (since, until),
        ).fetchone()["cnt"]

        # 有效片段数
        valid = conn.execute(
            "SELECT COUNT(*) as cnt FROM segments WHERE is_valid = 1 AND started_at >= ? AND started_at <= ?",
            (since, until),
        ).fetchone()["cnt"]

        # 总字数（有效片段）
        chars_row = conn.execute(
            "SELECT COALESCE(SUM(char_count), 0) as total_chars FROM segments WHERE is_valid = 1 AND started_at >= ? AND started_at <= ?",
            (since, until),
        ).fetchone()
        total_chars = chars_row["total_chars"]

        # 有效时长
        dur_row = conn.execute(
            "SELECT COALESCE(SUM(duration_s), 0) as total_dur FROM segments WHERE is_valid = 1 AND started_at >= ? AND started_at <= ?",
            (since, until),
        ).fetchone()
        total_duration = dur_row["total_dur"]

        # 今日分析次数
        analyses_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM analyses WHERE period_start >= ?",
            (since,),
        ).fetchone()["cnt"]

        return {
            "date": today_str,
            "total_segments": total,
            "valid_segments": valid,
            "total_chars": total_chars,
            "total_duration_s": round(total_duration, 1),
            "analyses_count": analyses_count,
        }
    finally:
        conn.close()
