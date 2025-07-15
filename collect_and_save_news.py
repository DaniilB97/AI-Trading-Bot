import asyncio
import sys
from pathlib import Path
import pandas as pd
import argparse

# Add project root to path to allow importing project modules
sys.path.append(str(Path(__file__).resolve().parent))

from main_files.config_loader import config
from eth_analysis_system import TelegramParser

logger = config.logger

async def main(output_path: str, days_back: int, limit_per_channel: int):
    """
    Collects news from Telegram channels configured in .env 
    and saves them to a CSV file.
    """
    if not config.validate_config():
        logger.error("❌ Configuration validation failed! Please check your .env file.")
        config.save_example_env()
        return

    telegram_cfg = config.get_telegram_config()
    channels = config.get_telegram_channels()

    if not channels:
        logger.error("No Telegram channels configured in .env file (TELEGRAM_CHANNELS).")
        return

    parser = TelegramParser(
        api_id=telegram_cfg['api_id'],
        api_hash=telegram_cfg['api_hash'],
        phone=telegram_cfg['phone']
    )

    logger.info(f"Attempting to connect to Telegram as {telegram_cfg['phone']}...")
    await parser.connect()
    logger.info("Successfully connected to Telegram.")

    logger.info(f"Fetching news from channels: {channels}")
    logger.info(f"Days back: {days_back}, Limit per channel: {limit_per_channel}")

    try:
        all_news_df = await parser.parse_multiple_channels(
            channel_ids=channels,
            days_back=days_back,
            limit=limit_per_channel
        )

        if all_news_df.empty:
            logger.warning("No news collected. The DataFrame is empty.")
        else:
            # Ensure the output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            all_news_df.to_csv(output_path, index=False, encoding='utf-8-sig') # utf-8-sig for better Excel compatibility
            logger.info(f"📰 Successfully collected {len(all_news_df)} news articles.")
            logger.info(f"💾 News saved to: {output_path}")

    except Exception as e:
        logger.error(f"An error occurred during news collection: {e}", exc_info=True)
    finally:
        logger.info("Closing Telegram connection...")
        await parser.close()
        logger.info("Telegram connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Telegram news and save to CSV.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/all_collected_news.csv",
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=config.get('analysis.days_back', 30), # Default from config or 30
        help="Number of days back to collect news from."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.get('telegram.parse_settings.messages_limit', 1000), # Default from config or 1000
        help="Maximum number of messages to fetch per channel."
    )
    args = parser.parse_args()

    asyncio.run(main(output_path=args.output, days_back=args.days, limit_per_channel=args.limit))
