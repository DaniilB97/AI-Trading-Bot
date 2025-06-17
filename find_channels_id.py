# find_channel_ids.py
import os
import asyncio
from dotenv import load_dotenv
from telethon import TelegramClient

load_dotenv()

API_ID = int(os.getenv('TELEGRAM_API_ID'))
API_HASH = os.getenv('TELEGRAM_API_HASH')
PHONE = os.getenv('TELEGRAM_PHONE')


async def main():
    client = TelegramClient('id_finder_session', API_ID, API_HASH)
    await client.start(phone=PHONE)
    print("Client Created and Connected. Fetching dialogs...\n")

    dialogs = await client.get_dialogs()

    print("Your Channels/Supergroups (and their IDs):")
    print("-------------------------------------------")
    for i, dialog in enumerate(dialogs):
        if dialog.is_channel:  # is_channel is true for channels and supergroups
            # For private channels without a username, dialog.entity.username will be None
            # dialog.name is the display name
            # dialog.id is the numerical ID (may be negative for channels/supergroups)
            print(f"{i + 1}. Name: \"{dialog.name}\"")
            print(f"   ID: {dialog.id}")
            if hasattr(dialog.entity, 'username') and dialog.entity.username:
                print(f"   Username: @{dialog.entity.username}")
            print("---")

    print("\nInstructions:")
    print("1. Identify your VIP signal channels from the list above.")
    print("2. Copy their 'ID' (the number, it might be negative).")
    print("3. Update your .env file. For example:")
    print("   VIP_TELEGRAM_CHANNELS=-1001234567890,-1009876543210")
    print("   (Use a comma to separate multiple IDs. Make sure to include the '-' if the ID is negative.)")

    await client.disconnect()


if __name__ == '__main__':
    asyncio.run(main())