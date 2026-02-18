import os
import sys

# Fix HOME environment variable for Windows/Playwright compatibility
# This is the critical fix for the "HOME environment variable is not set" error
if os.name == 'nt' and 'HOME' not in os.environ:
    user_profile = os.environ.get('USERPROFILE')
    if user_profile:
        os.environ['HOME'] = user_profile
        print(f"Configuration Fix: Set HOME environment variable to {user_profile}")
    else:
        print("Warning: USERPROFILE not found, cannot set HOME env var.")

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: Playwright not installed. Run 'pip install playwright'")
    sys.exit(1)

def run():
    print("Initializing Playwright...")
    with sync_playwright() as p:
        print("Launching Chromium browser (headless=False)...")
        try:
            # Launch browser in visible mode so the user can see it works
            browser = p.chromium.launch(headless=False, slow_mo=1000)
            context = browser.new_context()
            page = context.new_page()
            
            print("Navigating to https://www.google.com ...")
            page.goto("https://www.google.com")
            
            title = page.title()
            print(f"Success! Page loaded. Title: '{title}'")
            
            output_path = os.path.join(os.getcwd(), "chrome_verification_success.png")
            page.screenshot(path=output_path)
            print(f"Screenshot saved to: {output_path}")
            
            print("Closing browser...")
            browser.close()
            print("Verification Complete: Chrome is working correctly.")
        except Exception as e:
            print(f"Verification Failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    run()
