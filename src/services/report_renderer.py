# -*- coding: utf-8 -*-
"""
===================================
Report Engine - Jinja2 Report Renderer
===================================

Renders reports from Jinja2 templates. Falls back to caller's logic on template
missing or render error. Template path is relative to project root.
Any expensive data preparation should be injected by the caller via extra_context.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analyzer import AnalysisResult
from src.config import get_config
from src.market_phase_summary import format_public_market_status_line, format_public_phase_pack_excerpt
from src.services.decision_signal_summary import format_decision_signal_excerpt
from src.report_language import (
    get_localized_stock_name,
    get_report_labels,
    get_signal_level,
    get_chip_unavailable_reason,
    is_chip_structure_unavailable,
    localize_chip_health,
    localize_operation_advice,
    localize_trend_prediction,
    normalize_report_language,
)
from src.utils.data_processing import (
    normalize_model_used,
    signal_attribution_has_content,
    signal_attribution_weight_items,
)

logger = logging.getLogger(__name__)


def _escape_md(text: str) -> str:
    """Escape markdown special chars (*ST etc)."""
    if not text:
        return ""
    return text.replace("*", "\\*").replace("_", "\\_")


def _clean_sniper_value(val: Any) -> str:
    """Format sniper point value for display (strip label prefixes)."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).strip() if val else ""
    if not s or s == "N/A":
        return s or "N/A"
    prefixes = [
        "理想买入点：", "次优买入点：", "止损位：", "目标位：",
        "理想买入点:", "次优买入点:", "止损位:", "目标位:",
        "Ideal Entry:", "Secondary Entry:", "Stop Loss:", "Target:",
    ]
    for prefix in prefixes:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _resolve_templates_dir() -> Path:
    """Resolve template directory relative to project root."""
    config = get_config()
    base = Path(__file__).resolve().parent.parent.parent
    templates_dir = Path(config.report_templates_dir)
    if not templates_dir.is_absolute():
        return base / templates_dir
    return templates_dir


# ---------------------------------------------------------------------------
# Plain summary (busy non-professional readers)
# ---------------------------------------------------------------------------

# Action families used to decide "advice changed vs unchanged" between days.
# Ordered by risk priority: bearish wins in mixed text, hold before bullish so
# labels like "持有/轻仓低吸" classify conservatively.
_ADVICE_FAMILY_KEYWORDS = (
    ("sell", ("清仓", "卖出", "减仓", "sell", "reduce")),
    ("hold", ("持有", "hold")),
    ("buy", ("买入", "加仓", "建仓", "低吸", "增持", "buy", "add")),
    ("watch", ("观望", "等待", "回避", "wait", "watch")),
)


def advice_action_family(advice: Any) -> str:
    """Map an operation-advice label/text to an action family (sell/hold/buy/watch)."""
    text = str(advice or "").strip().lower()
    if not text:
        return ""
    for family, keywords in _ADVICE_FAMILY_KEYWORDS:
        for keyword in keywords:
            if keyword in text:
                return family
    return ""


def _shorten(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _build_plain_language(
    result: AnalysisResult,
    labels: Dict[str, str],
    report_language: str,
) -> Dict[str, str]:
    """Return the plain-language trio, synthesizing fallbacks when absent.

    The LLM-provided ``dashboard.core_conclusion.plain_language`` wins; missing
    keys degrade to position advice / stop-loss / risk-alert derived text so
    the plain summary never renders empty bullets.
    """
    dashboard = getattr(result, "dashboard", None) or {}
    core = dashboard.get("core_conclusion") or {}
    provided = core.get("plain_language")
    plain: Dict[str, str] = {}
    if isinstance(provided, dict):
        for key in ("action_now", "change_condition", "key_risk"):
            value = provided.get(key)
            if isinstance(value, str) and value.strip():
                plain[key] = _shorten(value.strip(), 60)

    if "action_now" not in plain:
        position_advice = core.get("position_advice") or {}
        action = position_advice.get("has_position") or position_advice.get("no_position")
        if not action:
            action = localize_operation_advice(result.operation_advice, report_language)
        plain["action_now"] = _shorten(action, 60)

    if "change_condition" not in plain:
        battle = dashboard.get("battle_plan") or {}
        sniper = battle.get("sniper_points") or {}
        stop_loss = _clean_sniper_value(sniper.get("stop_loss"))
        condition = ""
        if stop_loss and stop_loss not in ("N/A", "待补充"):
            condition = labels.get("plain_stop_loss_condition", "跌破 {stop_loss} 应止损离场").format(
                stop_loss=stop_loss
            )
        else:
            phase_decision = dashboard.get("phase_decision") or {}
            watch_conditions = phase_decision.get("watch_conditions") or []
            if watch_conditions and isinstance(watch_conditions, list):
                condition = _shorten(watch_conditions[0], 60)
        if condition:
            plain["change_condition"] = _shorten(condition, 60)

    if "key_risk" not in plain:
        intel = dashboard.get("intelligence") or {}
        risk_alerts = intel.get("risk_alerts") or []
        risk = risk_alerts[0] if isinstance(risk_alerts, list) and risk_alerts else ""
        if not risk:
            risk = getattr(result, "risk_warning", "") or ""
            risk = re.split(r"[；;。]", risk)[0] if risk else ""
        if risk:
            plain["key_risk"] = _shorten(risk, 50)

    return plain


def render(
    platform: str,
    results: List[AnalysisResult],
    report_date: Optional[str] = None,
    summary_only: bool = False,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Render report using Jinja2 template.

    Args:
        platform: One of: markdown, wechat, brief
        results: List of AnalysisResult
        report_date: Report date string (default: today)
        summary_only: Whether to output summary only
        extra_context: Additional template context

    Returns:
        Rendered string, or None on error (caller should fallback).
    """
    from datetime import datetime

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        logger.warning("jinja2 not installed, report renderer disabled")
        return None

    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    templates_dir = _resolve_templates_dir()
    template_name = f"report_{platform}.j2"
    template_path = templates_dir / template_name
    if not template_path.exists():
        logger.debug("Report template not found: %s", template_path)
        return None

    report_language = normalize_report_language(
        (extra_context or {}).get("report_language")
        or next(
            (getattr(result, "report_language", None) for result in results if getattr(result, "report_language", None)),
            None,
        )
        or getattr(get_config(), "report_language", "zh")
    )
    labels = get_report_labels(report_language)

    # Plain-summary inputs (busy-reader mode); fail-open defaults keep legacy shape
    plain_summary = bool((extra_context or {}).get("plain_summary"))
    previous_advice_by_code = (extra_context or {}).get("previous_advice_by_code") or {}
    if not isinstance(previous_advice_by_code, dict):
        previous_advice_by_code = {}

    # Build template context with pre-computed signal levels (sorted by score)
    sorted_results = sorted(results, key=lambda x: x.sentiment_score, reverse=True)
    sorted_enriched = []
    for r in sorted_results:
        st, se, _ = get_signal_level(r.operation_advice, r.sentiment_score, report_language)
        rn = get_localized_stock_name(r.name, r.code, report_language)
        previous_advice_raw = previous_advice_by_code.get(r.code, "")
        previous_family = advice_action_family(previous_advice_raw)
        current_family = advice_action_family(r.operation_advice)
        sorted_enriched.append({
            "result": r,
            "signal_text": st,
            "signal_emoji": se,
            "stock_name": _escape_md(rn),
            "localized_operation_advice": localize_operation_advice(r.operation_advice, report_language),
            "localized_trend_prediction": localize_trend_prediction(r.trend_prediction, report_language),
            "plain": _build_plain_language(r, labels, report_language) if plain_summary else {},
            "previous_advice": (
                _shorten(localize_operation_advice(previous_advice_raw, report_language), 12)
                if previous_family
                else ""
            ),
            # 无历史记录（首次分析/查询失败）按"有变"处理，宁多看一眼不漏
            "advice_changed": (not previous_family) or previous_family != current_family,
        })

    buy_count = sum(1 for r in results if getattr(r, "decision_type", "") == "buy")
    sell_count = sum(1 for r in results if getattr(r, "decision_type", "") == "sell")
    hold_count = sum(1 for r in results if getattr(r, "decision_type", "") in ("hold", ""))
    show_llm_model = bool(getattr(get_config(), "report_show_llm_model", True))
    models_used: List[str] = []
    if show_llm_model:
        for result in results:
            model = normalize_model_used(getattr(result, "model_used", None))
            if model:
                models_used.append(model)
        models_used = list(dict.fromkeys(models_used))

    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def failed_checks(checklist: List[str]) -> List[str]:
        return [c for c in (checklist or []) if c.startswith("❌") or c.startswith("⚠️")]

    def phase_pack_excerpt(result: AnalysisResult) -> str:
        return format_public_phase_pack_excerpt(
            getattr(result, "market_phase_summary", None),
            getattr(result, "analysis_context_pack_overview", None),
            source=getattr(result, "analysis_visibility_source", None) or "evaluator_snapshot",
            report_language=report_language,
        )

    def decision_signal_excerpt(result: AnalysisResult) -> str:
        return format_decision_signal_excerpt(
            getattr(result, "decision_signal_summary", None),
            report_language=report_language,
        )

    def market_status_line() -> str:
        for source_results in (results or [], sorted_results):
            for result in source_results:
                line = format_public_market_status_line(
                    getattr(result, "market_phase_summary", None),
                    report_language=report_language,
                )
                if line:
                    return line
        return ""

    context: Dict[str, Any] = {
        "report_date": report_date,
        "report_timestamp": report_timestamp,
        "results": sorted_results,
        "enriched": sorted_enriched,  # Sorted by sentiment_score desc
        "summary_only": summary_only,
        "plain_summary": plain_summary,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "labels": labels,
        "report_language": report_language,
        "models_used": models_used,
        "show_llm_model": show_llm_model,
        "market_status_line": market_status_line(),
        "escape_md": _escape_md,
        "clean_sniper": _clean_sniper_value,
        "failed_checks": failed_checks,
        "phase_pack_excerpt": phase_pack_excerpt,
        "decision_signal_excerpt": decision_signal_excerpt,
        "history_by_code": {},
        "get_chip_unavailable_reason": get_chip_unavailable_reason,
        "is_chip_structure_unavailable": is_chip_structure_unavailable,
        "localize_operation_advice": localize_operation_advice,
        "localize_trend_prediction": localize_trend_prediction,
        "localize_chip_health": localize_chip_health,
        "signal_attribution_has_content": signal_attribution_has_content,
        "signal_attribution_weight_items": signal_attribution_weight_items,
    }
    if extra_context:
        safe_extra_context = dict(extra_context)
        safe_extra_context.pop("labels", None)
        safe_extra_context.pop("report_language", None)
        context.update(safe_extra_context)

    try:
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(default=False),
        )
        template = env.get_template(template_name)
        return template.render(**context)
    except Exception as e:
        logger.warning("Report render failed for %s: %s", template_name, e)
        return None
