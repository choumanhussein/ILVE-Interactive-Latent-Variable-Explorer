"""
Setup script for Weights & Biases integration.

Run this once to configure W&B for the project.
"""

import wandb
import sys
import os


def setup_wandb():
    """Setup Weights & Biases for the project."""
    print("🔧 Setting up Weights & Biases...")
    print("=" * 40)
    
    print("1. If you don't have a W&B account, create one at: https://wandb.ai/")
    print("2. Find your API key at: https://wandb.ai/authorize")
    print("3. This script will help you login")
    print()
    
    try:
        # Try to login
        wandb.login()
        print("✅ Successfully logged into Weights & Biases!")
        
        # Test a quick run
        print("\n🧪 Testing W&B integration...")
        test_run = wandb.init(
            project="vae-test", 
            name="setup-test",
            tags=["setup", "test"]
        )
        
        # Log a simple metric
        wandb.log({"test_metric": 1.0})
        
        # Finish the test run
        wandb.finish()
        
        print("✅ W&B test successful!")
        print("\n🎉 Setup complete! You can now run training with W&B logging.")
        
    except Exception as e:
        print(f"❌ W&B setup failed: {e}")
        print("\n💡 Alternative: You can run training with --no-wandb to skip W&B logging")
        return False
    
    return True


def check_wandb_status():
    """Check if W&B is properly configured."""
    try:
        # Check if logged in
        if wandb.api.api_key is None:
            print("❌ W&B not configured. Run 'python setup_wandb.py' to set up.")
            return False
        else:
            print("✅ W&B is properly configured!")
            return True
    except:
        print("❌ W&B not available. Install with: pip install wandb")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        check_wandb_status()
    else:
        setup_wandb()