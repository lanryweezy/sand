import os
import sys
from playwright.sync_api import sync_playwright

# Fix for "HOME environment variable is not set" on Windows
if os.name == 'nt' and 'HOME' not in os.environ:
    user_profile = os.environ.get('USERPROFILE')
    if user_profile:
        os.environ['HOME'] = user_profile

def browse(url):
    print(f"Initializing browser to visit: {url}")
    with sync_playwright() as p:
        # Launch visible browser
        browser = p.chromium.launch(headless=False, slow_mo=1000)
        context = browser.new_context()
        page = context.new_page()

        try:
            print(f"Navigating to {url}...")
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            
            title = page.title()
            print(f"Page Title: {title}")
            
            # Take a screenshot
            domain = url.split("//")[-1].split("/")[0].replace(".", "_")
            screenshot_path = os.path.join(os.getcwd(), f"screenshot_{domain}.png")
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved: {screenshot_path}")
            
            # Keep browser open for a few seconds if running interactively
            page.wait_for_timeout(3000)

        except Exception as e:
            print(f"Error visiting {url}: {e}")
        finally:
            browser.close()
            print("Browser closed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python browse.py <URL>")
        # Default behavior for demonstration
        default_url = "https://www.google.com"
        print(f"No URL provided. Defaulting to: {default_url}")
        browse(default_url)
    else:
        browse(sys.argv[1])
