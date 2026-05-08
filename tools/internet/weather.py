"""Tool: Current weather and short forecast via Open-Meteo."""
from __future__ import annotations

import logging
from typing import Any

from adapters.base.platform_adapter import PlatformAdapter
from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


def _wmo_condition(code: int | None) -> str:
    if code is None:
        return "Unknown"
    if code == 0:
        return "Clear"
    if 1 <= code <= 3:
        return "Partly cloudy / overcast"
    if 45 <= code <= 48:
        return "Foggy"
    if 51 <= code <= 55:
        return "Drizzle"
    if 61 <= code <= 65:
        return "Rain"
    if 71 <= code <= 75:
        return "Snow"
    if 80 <= code <= 82:
        return "Showers"
    if 95 <= code <= 99:
        return "Thunderstorm"
    return "Other"


class WeatherTool(BaseTool):
    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get current weather for a city. Returns temperature, conditions, and forecast."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        }

    async def execute(
        self, params: dict[str, Any], adapter: PlatformAdapter
    ) -> ToolResult:
        city = (params.get("city") or "").strip()
        if not city:
            return ToolResult(success=False, message="No city provided")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                geo_resp = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": city, "count": 1},
                )
                geo_resp.raise_for_status()
                geo = geo_resp.json()
                results = geo.get("results") or []
                if not results:
                    return ToolResult(
                        success=False,
                        message=f"No location found for '{city}'.",
                    )

                place = results[0]
                lat = place["latitude"]
                lon = place["longitude"]
                label = place.get("name", city)

                fc_resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,weather_code",
                        "daily": "temperature_2m_max,temperature_2m_min",
                        "timezone": "auto",
                        "forecast_days": 2,
                    },
                )
                fc_resp.raise_for_status()
                fc = fc_resp.json()

            current = fc.get("current") or {}
            temp = current.get("temperature_2m")
            wcode = current.get("weather_code")
            if isinstance(wcode, float):
                wcode = int(wcode)
            condition = _wmo_condition(int(wcode) if wcode is not None else None)

            daily = fc.get("daily") or {}
            tmax = daily.get("temperature_2m_max") or []
            tmin = daily.get("temperature_2m_min") or []

            def _day(idx: int) -> tuple[float | None, float | None]:
                hi = tmax[idx] if idx < len(tmax) else None
                lo = tmin[idx] if idx < len(tmin) else None
                return hi, lo

            max0, min0 = _day(0)
            max1, min1 = _day(1)

            def fmt(v: float | None) -> str:
                return f"{v:.0f}" if isinstance(v, (int, float)) else "?"

            msg = (
                f"Currently {fmt(temp)}°C in {label}, {condition}. "
                f"Today's high: {fmt(max0)}°C, low: {fmt(min0)}°C. "
                f"Tomorrow: {fmt(max1)}°C / {fmt(min1)}°C."
            )

            return ToolResult(
                success=True,
                message=msg,
                data={"speak_result": True},
            )
        except Exception as e:
            logger.exception("weather failed for city=%r", city)
            return ToolResult(
                success=False,
                message=f"Weather lookup failed: {e}",
            )
