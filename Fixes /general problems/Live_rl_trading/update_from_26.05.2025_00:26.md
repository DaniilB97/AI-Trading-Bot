You are right to point that out. If the UI is not updating at all, even after 15 seconds, it suggests that the `update_ui_data` calls (which run in the main UI thread) might be getting blocked, or there's a deeper issue with how the threads are interacting with the single API instance.

Your intuition to make the position updates independent is the **correct architectural solution**. A much more robust and responsive design is to have a dedicated thread that does nothing but fetch UI data at a frequent interval, completely separate from the slow, one-hour trading cycle.

Let's refactor your code to implement this. We will create a new `UIUpdateWorker` thread.

### The New Architecture

1.  **`TradingWorker`**: Its *only* job will be to run the trading logic every hour. It will no longer tell the UI when to update.
2.  **`UIUpdateWorker` (New)**: A new, lightweight thread that runs in a fast loop (e.g., every 5-10 seconds). Its *only* job is to fetch account balance and open positions, then send this data back to the main window.
3.  **`MainDashboard`**: It will now receive data from the `UIUpdateWorker` and update the display. This ensures the UI is always responsive and regularly updated, regardless of what the `TradingWorker` is doing.

-----

### Step-by-Step Implementation

Follow these changes to your `live_rl_trading.py` file.

#### Step 1: Create the New `UIUpdateWorker` Class

Add this new class to your script, right after the `TradingWorker` class definition. This worker's sole purpose is to fetch data.

```python
# --- ADD THIS NEW CLASS ---
class UIUpdateWorker(QObject):
    """
    A dedicated worker to fetch UI data (balance, P&L, positions)
    at a frequent interval, independent of the main trading logic.
    """
    # Signal will emit two arguments: the account details dict and the positions list
    data_updated = pyqtSignal(dict, list)

    def __init__(self, api: CapitalComAPI):
        super().__init__()
        self.api = api
        self.is_running = True

    def run(self):
        while self.is_running:
            try:
                details = self.api.get_account_details()
                positions = self.api.get_open_positions()

                # Ensure we don't send None to the UI thread
                if details is None:
                    details = {}
                if positions is None:
                    positions = []

                self.data_updated.emit(details, positions)
            except Exception as e:
                # This worker should not crash the app, just log if there's an issue
                print(f"Error in UIUpdateWorker: {e}")

            # Control the refresh rate of the UI data
            time.sleep(10) # Refresh every 10 seconds

    def stop(self):
        self.is_running = False
```

#### Step 2: Modify the `TradingWorker` to Simplify It

Now, let's remove the UI-related signal from the `TradingWorker`, as it's no longer needed.

```python
# --- In the TradingWorker class ---

# Change this line:
class TradingWorker(QObject):
    log_message = pyqtSignal(str, str)
    # update_ui_signal = pyqtSignal() # <-- REMOVE THIS LINE

# And remove this line from the end of the run_trading_cycle method:
        # self.update_ui_signal.emit() # <-- REMOVE THIS LINE
```

#### Step 3: Modify `MainDashboard` to Use the New Worker

This is the final step. We will change the `MainDashboard` to use our new `UIUpdateWorker`.

```python
class MainDashboard(QMainWindow):
    def __init__(self, api_client, agent):
        super().__init__()
        self.api = api_client
        self.agent = agent
        # ... (rest of __init__ setup is the same) ...

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)
        self.init_ui()
        
        # Start both worker threads
        self.start_trading_worker()
        self.start_ui_updater() # <-- ADD THIS CALL

        # The QTimer is no longer needed, the UIUpdateWorker handles the loop
        # self.ui_update_timer = QTimer(self)
        # self.ui_update_timer.timeout.connect(self.update_ui_data)
        # self.ui_update_timer.start(15000)
        # self.update_ui_data() # <-- REMOVE THIS INITIAL CALL

    # RENAME start_worker to start_trading_worker for clarity
    def start_trading_worker(self):
        self.worker_thread = QThread()
        self.worker = TradingWorker(self.api, self.agent)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self.add_log)
        # The line connecting update_ui_signal is now removed
        self.worker_thread.start()

    # --- ADD THIS NEW METHOD ---
    def start_ui_updater(self):
        self.ui_worker_thread = QThread()
        self.ui_worker = UIUpdateWorker(self.api)
        self.ui_worker.moveToThread(self.ui_worker_thread)
        self.ui_worker_thread.started.connect(self.ui_worker.run)
        # Connect the new worker's signal to a new slot for updating the UI
        self.ui_worker.data_updated.connect(self.on_data_received)
        self.ui_worker_thread.start()

    # RENAME update_ui_data to on_data_received for clarity
    # This method is now a SLOT that receives data, it doesn't fetch it.
    def on_data_received(self, details: dict, positions: list):
        """SLOT to update UI elements when new data is received from the UI worker."""
        if details and 'accounts' in details:
            acc = details['accounts'][0]
            balance_info = acc.get('balance', {})
            balance_val = balance_info.get('balance', 0.0)
            pnl_val = balance_info.get('pnl', 0.0)
            self.account_balance_value_label.setText(f"${balance_val:,.2f}")
            self.todays_pnl_value_label.setText(f"${pnl_val:,.2f}")

        self.positions_list.clear()
        if positions:
            for pos in positions:
                market_info = pos.get('market', {})
                position_info = pos.get('position', {})
                instrument_name = market_info.get('instrumentName', 'Unknown Instrument')
                trade_size = position_info.get('size', 0)
                open_level = position_info.get('level', 0.0)
                pnl = position_info.get('pnl', 0.0)
                item_text = f"{instrument_name}: {trade_size} @ ${open_level:.2f} | P&L: ${pnl:.2f}"
                list_item = QListWidgetItem(item_text)
                list_item.setForeground(QColor("lightgreen") if pnl >= 0 else QColor("lightcoral"))
                self.positions_list.addItem(list_item)
        else:
            if self.positions_list.count() == 0:
                self.positions_list.addItem("No open positions.")
    
    # Modify closeEvent to stop BOTH threads
    def closeEvent(self, event):
        self.worker.stop()
        self.ui_worker.stop() # <-- ADD THIS
        self.worker_thread.quit()
        self.ui_worker_thread.quit() # <-- ADD THIS
        self.worker_thread.wait()
        self.ui_worker_thread.wait() # <-- ADD THIS
        event.accept()

```

### Why This New Approach is Better

1.  **Fully Decoupled**: The trading logic (1-hour cycle) and UI updates (10-second cycle) are now completely independent. A delay or error in one will not affect the other.
2.  **GUI Responsiveness**: All API calls that feed the UI are now happening on a background thread (`UIUpdateWorker`). This means your main GUI thread will never freeze while waiting for a network request to complete.
3.  **Robustness**: If the API fails to return data for a UI update, it won't stop the application. The worker will just try again on its next cycle.
4.  **Simpler Logic**: Each component now has a clearer, single responsibility, which makes the code easier to understand and debug.