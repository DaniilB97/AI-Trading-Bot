# -*- coding: utf-8 -*-
# test_trade_execution.py
import os
import time
import logging
from dotenv import load_dotenv

# Import your Capital.com API helper
from capital_request import CapitalComAPI

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_EPIC = "GOLD"
TEST_TRADE_SIZE = 0.01

def run_test():
    """
    Runs a simple, step-by-step test to open and close a position.
    """
    logger.info("--- Starting Capital.com Trade Execution Test ---")

    # 1. Initialize and Login
    api_key = os.getenv("CAPITAL_API_KEY")
    identifier = os.getenv("CAPITAL_IDENTIFIER")
    password = os.getenv("CAPITAL_PASSWORD")

    if not all([api_key, identifier, password]):
        logger.error("❌ Missing API credentials in .env file. Test cannot proceed.")
        return

    api = CapitalComAPI(api_key, identifier, password)
    if not api.login_and_get_tokens():
        logger.error("❌ Login failed. Aborting test.")
        return
    logger.info("✅ Step 1: Login Successful.")

    # 2. Check for existing positions
    logger.info("\n--- Step 2: Checking for existing positions... ---")
    open_positions = api.get_open_positions()
    if open_positions is None:
        logger.error("❌ Failed to get open positions.")
        return
        
    position_for_epic = next((p for p in open_positions if p.get('market',{}).get('epic') == TEST_EPIC), None)

    if position_for_epic:
        logger.warning(f"⚠️ Found an existing position for {TEST_EPIC}. Attempting to close it first.")
        deal_id = position_for_epic.get('position', {}).get('dealId')
        close_result = api.close_position(deal_id)
        if close_result:
            logger.info(f"✅ Successfully closed existing position {deal_id}.")
            time.sleep(5) # Wait a moment for the account to update
        else:
            logger.error(f"❌ Failed to close existing position {deal_id}. Please close it manually and retry. Aborting test.")
            return

    # 3. Open a new position
    logger.info(f"\n--- Step 3: Attempting to OPEN a new BUY position for {TEST_EPIC} ---")
    create_result = api.create_position(epic=TEST_EPIC, direction="BUY", size=TEST_TRADE_SIZE)

    if create_result and create_result.get('dealReference'):
        logger.info(f"✅ Position creation request sent successfully. Deal Reference: {create_result['dealReference']}")
    else:
        logger.error("❌ Failed to create a new position. Aborting test.")
        return
    
    # 4. Verify the position was opened
    logger.info("\n--- Step 4: Verifying that the position is open... ---")
    time.sleep(5) # Wait a few seconds for the order to be processed
    
    open_positions = api.get_open_positions()
    new_position = next((p for p in open_positions if p.get('market',{}).get('epic') == TEST_EPIC), None)

    if not new_position:
        logger.error("❌ Verification failed. The new position was not found.")
        return
    
    deal_id_to_close = new_position.get('position', {}).get('dealId')
    logger.info(f"✅ Verification successful! Found open position with Deal ID: {deal_id_to_close}")

    # 5. Close the new position
    logger.info(f"\n--- Step 5: Attempting to CLOSE the new position... ---")
    close_result = api.close_position(deal_id_to_close)
    
    if close_result:
        logger.info(f"✅ Successfully closed new position {deal_id_to_close}.")
    else:
        logger.error(f"❌ Failed to close the new position. Please check your account manually.")
        return

    logger.info("\n🎉 --- Trade Execution Test Completed Successfully! --- 🎉")


if __name__ == "__main__":
    run_test()

