from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def capture_screenshot(url, output_path):
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure it runs without opening a window
    chrome_options.add_argument("window-size=1200x600")  # Default window size

    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)

    # Load the URL
    driver.get(url)

    # Save the screenshot
    driver.save_screenshot(output_path)
    driver.quit()

# Use this function to capture a screenshot of your HTML
capture_screenshot('path', 'output_html_screenshot.png')
