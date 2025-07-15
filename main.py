"""
ETH Analysis System - Main Entry Point
Updated to use configuration from .env and config.yaml
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from main_files.config_loader import config
from eth_analysis_system import ETHAnalysisSystem
import logging

# Get logger from config
logger = config.logger

class ETHAnalysisRunner:
    """Main runner for ETH analysis system"""
    
    def __init__(self):
        # Validate configuration
        if not config.validate_config():
            raise ValueError("Invalid configuration. Please check your .env file")
        
        # Initialize system with config
        self.system = ETHAnalysisSystem(config.get_telegram_config())
        self.channels = config.get_telegram_channels()
        self.logger = logger
        
    async def run_analysis(self):
        """Run complete analysis pipeline"""
        try:
            self.logger.info("🚀 Starting ETH Analysis System")
            self.logger.info(f"📱 Monitoring channels: {self.channels}")
            
            # Run analysis
            results = await self.system.run_full_analysis(self.channels)
            
            # Save results
            self._save_results(results)
            
            # Send notifications if enabled
            if config.get('notifications.enabled'):
                await self._send_notifications(results)
            
            # Display summary
            self._display_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {e}")
            raise
    
    def _save_results(self, results):
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save report
        report_path = f"reports/eth_analysis_{timestamp}.json"
        Path("reports").mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(results['report'], f, indent=2, default=str)
        
        self.logger.info(f"📄 Report saved to {report_path}")
        
        # Save chart if exists
        if 'chart' in results:
            chart_path = f"reports/eth_chart_{timestamp}.html"
            results['chart'].write_html(chart_path)
            self.logger.info(f"📊 Chart saved to {chart_path}")
    
    async def _send_notifications(self, results):
        """Send notifications based on triggers"""
        trading_signals = results['report'].get('trading_signals', {}) # Use .get for safety
        
        # Check notification triggers
        should_notify = False
        notification_text = []
        
        # High confidence signal
        if trading_signals.get('confidence', 0) >= config.get('analysis.confidence_threshold'): # Use .get
            should_notify = True
            notification_text.append(
                f"🎯 Strong {trading_signals.get('action', 'unknown').upper()} signal detected! " # Use .get and provide default
                f"Confidence: {trading_signals.get('confidence', 0)*100:.1f}%"
            )
        
        # High impact news
        if 'news_impact' in results['report']:
            high_impact_count = results['report']['news_impact']['high_impact_count']
            if high_impact_count > 0:
                should_notify = True
                notification_text.append(
                    f"📰 {high_impact_count} high impact news detected!"
                )
        
        # Price prediction
        if 'predictions' in results['report'] and results['report']['predictions']:
            predictions = results['report']['predictions']['predictions']
            current_price = results['report']['current_analysis']['price']
            week_prediction = predictions[-1]
            change_pct = ((week_prediction - current_price) / current_price) * 100
            
            if abs(change_pct) >= 5.0:
                should_notify = True
                notification_text.append(
                    f"📈 Significant price movement predicted: {change_pct:+.1f}% in 7 days"
                )
        
        if should_notify and notification_text:
            await self._send_webhook_notification('\n'.join(notification_text))
    
    async def _send_webhook_notification(self, message: str):
        """Send notification to webhook"""
        webhook_url = config.get('notifications.webhook')
        if not webhook_url:
            self.logger.warning("Webhook URL not configured")
            return
        
        # Implementation depends on webhook type (Slack, Discord, etc.)
        # This is a placeholder
        self.logger.info(f"📢 Notification: {message}")
    
    def _display_summary(self, results):
        """Display analysis summary"""
        report = results['report']
        # current = report.get('current_analysis', {}) # 'current_analysis' key does not exist in report
        trading_signals = report.get('trading_signals', {}) 
        
        current_price_val = report.get('current_price')
        technical_summary = report.get('technical_summary', {})
        trend_val = technical_summary.get('trend', {}).get('short_term', 'N/A') # Example: using short-term trend

        print("\n" + "="*60)
        print("📊 ETH ANALYSIS SUMMARY")
        print("="*60)
        
        if current_price_val is not None:
            print(f"\n💰 Current Price: ${current_price_val:.2f}")
        else:
            print("\n💰 Current Price: N/A")
            
        # 'change_24h' is not directly in the report, needs calculation or to be added to report.
        # For now, display N/A or calculate if possible from price_df if it were passed here.
        # Simplest for now is N/A.
        print(f"📈 24h Change: N/A") # Placeholder, as 24h change is not in the current report structure
        print(f"📊 Trend: {trend_val}")
        
        print(f"\n🎯 RECOMMENDATION: {trading_signals.get('action', 'N/A').upper()}")
        print(f"💪 Confidence: {trading_signals.get('confidence', 0)*100:.1f}%")
        
        print("\n📝 Reasons:")
        for reason in trading_signals.get('reasons', []): # Use .get
            print(f"  • {reason}")
        
        if 'predictions' in report and report.get('predictions'): # Use .get
            print("\n🔮 7-Day Price Prediction:")
            # current_price was from the old 'current' dict. Use current_price_val from report.
            if current_price_val is not None:
                # Corrected loop variable name from 'price' to 'price_pred_val'
                for date, price_pred_val in zip( 
                    report['predictions']['dates'][-3:],  # Last 3 days
                    report['predictions']['predictions'][-3:]
                ):
                    change = (price_pred_val - current_price_val) / current_price_val * 100
                    print(f"  {date}: ${price_pred_val:.2f} ({change:+.1f}%)")
            else:
                print("  Price prediction cannot be displayed as current price is unavailable.")
        
        print("\n" + "="*60)

async def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════╗
    ║   ETH Analysis & Prediction System    ║
    ║         Powered by AI & ML            ║
    ╚═══════════════════════════════════════╝
    """)
    
    try:
        # Create runner
        runner = ETHAnalysisRunner()
        
        # Run analysis
        await runner.run_analysis()
        
        print("\n✅ Analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Fatal error in main")
        sys.exit(1)

def run_scheduled():
    """Run analysis on schedule"""
    import schedule
    import time
    
    def job():
        asyncio.run(main())
    
    # Schedule runs
    schedule.every().day.at("09:00").do(job)  # Morning analysis
    schedule.every().day.at("21:00").do(job)  # Evening analysis
    
    logger.info("⏰ Scheduled analysis started")
    logger.info("   Morning run: 09:00")
    logger.info("   Evening run: 21:00")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ETH Analysis System')
    parser.add_argument('--scheduled', action='store_true', 
                       help='Run on schedule instead of once')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    if args.scheduled:
        run_scheduled()
    else:
        asyncio.run(main())
