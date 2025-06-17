#!/usr/bin/env python3
"""
Setup script for ETH Analysis System
Helps with initial configuration and dependency installation
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

class SystemSetup:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        
    def check_python_version(self):
        """Check Python version"""
        print("🐍 Checking Python version...")
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        print(f"✅ Python {sys.version.split()[0]} detected")
        return True
    
    def create_virtualenv(self):
        """Create virtual environment"""
        print("\n📦 Creating virtual environment...")
        venv_path = self.root_dir / "venv"
        
        if venv_path.exists():
            response = input("Virtual environment exists. Recreate? (y/n): ")
            if response.lower() != 'y':
                return True
            shutil.rmtree(venv_path)
        
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created")
        
        # Show activation command
        if os.name == 'nt':  # Windows
            activate_cmd = "venv\\Scripts\\activate"
        else:  # Unix
            activate_cmd = "source venv/bin/activate"
        
        print(f"\n💡 To activate: {activate_cmd}")
        return True
    
    def install_dependencies(self):
        """Install required packages"""
        print("\n📚 Installing dependencies...")
        
        # Check if in virtual environment
        if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
            print("⚠️  Not in virtual environment. It's recommended to activate venv first.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # Install TA-Lib first (special handling)
        print("\n📊 Installing TA-Lib...")
        if os.name == 'nt':  # Windows
            print("⚠️  TA-Lib requires manual installation on Windows")
            print("   Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
            print("   Then: pip install downloaded_file.whl")
        else:
            print("   You may need to install TA-Lib C library first:")
            print("   Ubuntu/Debian: sudo apt-get install ta-lib")
            print("   macOS: brew install ta-lib")
        
        # Install other requirements
        print("\n📦 Installing Python packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencies installed")
        return True
    
    def setup_config_files(self):
        """Setup configuration files"""
        print("\n⚙️  Setting up configuration...")
        
        # Create .env from example
        env_example = self.root_dir / ".env.example"
        env_file = self.root_dir / ".env"
        
        if not env_file.exists():
            if env_example.exists():
                shutil.copy(env_example, env_file)
                print("✅ Created .env from .env.example")
            else:
                # Create minimal .env
                self._create_minimal_env()
                print("✅ Created minimal .env file")
        else:
            print("ℹ️  .env already exists")
        
        # Check config.yaml
        config_file = self.root_dir / "config.yaml"
        if not config_file.exists():
            print("⚠️  config.yaml not found. Using defaults.")
        
        print("\n📝 Please edit .env file with your credentials:")
        print("   - TELEGRAM_API_ID")
        print("   - TELEGRAM_API_HASH")
        print("   - TELEGRAM_PHONE")
        print("   - TELEGRAM_CHANNELS")
        
        return True
    
    def _create_minimal_env(self):
        """Create minimal .env file"""
        content = """# Telegram API credentials
TELEGRAM_API_ID=YOUR_API_ID
TELEGRAM_API_HASH=YOUR_API_HASH
TELEGRAM_PHONE=+YOUR_PHONE

# Channels to monitor
TELEGRAM_CHANNELS=@eth_news,@crypto_signals

# Model settings
MODEL_EPOCHS=50
MODEL_BATCH_SIZE=32

# Analysis settings
ANALYSIS_DAYS_BACK=30
"""
        with open(".env", "w") as f:
            f.write(content)
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directories...")
        
        dirs = [
            "reports",
            "logs",
            "models",
            "cache",
            "data"
        ]
        
        for dir_name in dirs:
            dir_path = self.root_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"   ✅ {dir_name}/")
        
        return True
    
    def check_telegram_setup(self):
        """Guide for Telegram API setup"""
        print("\n📱 Telegram API Setup:")
        print("1. Go to https://my.telegram.org")
        print("2. Log in with your phone number")
        print("3. Go to 'API development tools'")
        print("4. Create a new application")
        print("5. Copy API ID and API Hash to .env file")
        print("\n💡 Tips:")
        print("   - Use a dedicated phone number if possible")
        print("   - Keep your API credentials secure")
        print("   - Don't share session files")
        
        return True
    
    def run_setup(self):
        """Run complete setup"""
        print("""
╔════════════════════════════════════════╗
║     ETH Analysis System Setup          ║
╚════════════════════════════════════════╝
        """)
        
        steps = [
            ("Check Python version", self.check_python_version),
            ("Create virtual environment", self.create_virtualenv),
            ("Install dependencies", self.install_dependencies),
            ("Setup configuration", self.setup_config_files),
            ("Create directories", self.create_directories),
            ("Telegram setup guide", self.check_telegram_setup)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*40}")
            print(f"Step: {step_name}")
            print('='*40)
            
            if not step_func():
                print(f"\n❌ Setup failed at: {step_name}")
                return False
        
        print("\n" + "="*50)
        print("✅ Setup completed successfully!")
        print("="*50)
        print("\nNext steps:")
        print("1. Edit .env file with your credentials")
        print("2. Activate virtual environment:")
        if os.name == 'nt':
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("3. Run the system:")
        print("   python main.py")
        print("\nFor scheduled runs:")
        print("   python main.py --scheduled")
        
        return True

def main():
    setup = SystemSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()