# mcp-weather-server.py

from mcp.server import MCPServer

mcp = MCPServer("weather")


@mcp.tool()
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""

    weather_data = {
        "San Francisco": {
            "temperature": 65,
            "unit": "F",
            "condition": "Partly cloudy",
        },
        "New York": {
            "temperature": 72,
            "unit": "F",
            "condition": "Sunny",
        },
        "Seattle": {
            "temperature": 58,
            "unit": "F",
            "condition": "Rainy",
        },
    }

    return weather_data.get(
        city,
        {
            "temperature": None,
            "unit": "F",
            "condition": "Unknown",
        },
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")