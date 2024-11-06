import os
import csv
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

from Logger import LOGGER
logger = LOGGER

class Scrape:
    def setup_filepath(self):
        filePath = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(filePath):
            os.mkdir(filePath)

        file_path = os.path.join(filePath, f"Calendar_Event.csv")

        if not os.path.exists(file_path):
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                writer.writerow(["date", "title", "actual", "forecast", "previous"])

        return file_path


    def get_endpoints(self):
        start_date = datetime(2007, 1, 1)

        end_date = datetime(2022, 11, 14)

        endpoints = []

        current_date = start_date
        while current_date <= end_date:
            endpoint = current_date.strftime("%b%d.%Y").lower()
            endpoints.append(endpoint)
            
            current_date += timedelta(days=1)

        return endpoints



    def scrape_calenda_events(self, endpoints, file_path, events):
        options = Options()
        # options.add_argument("--headless")
        options.set_preference("profile.managed_default_content_settings.images", 2)
        driver = webdriver.Firefox(options=options)


        for endpoint in endpoints:
            driver.get(f"https://www.energyexch.com/calendar?day={endpoint}")

            element = driver.find_element(By.CLASS_NAME, 'calendar__table')

            html_content = element.get_attribute('innerHTML')
            soup = BeautifulSoup(html_content, 'html.parser')
            table_row = soup.select('.calendar__row ')

            data = []
            for row in table_row:
                try:
                    if row.select_one('.calendar__event-title').get_text(strip=True) in events:
                        date = endpoint
                        title = row.select_one('.calendar__event-title').get_text(strip=True)
                        actual = row.select_one('.calendar__actual').get_text(strip=True)
                        forecast = row.select_one('.calendar__forecast').get_text(strip=True)
                        previous = row.select_one('.calendar__previous').get_text(strip=True)
                        data = [date, title,  actual, forecast, previous]

                        with open(file_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerows([data])                
                        
                        logger.info(f"Successfully scraped {title} for Date: {date}")

                except Exception as e:
                    logger.error(f"Error while scraping {e}")




if __name__ == "__main__":
    scraper = Scrape()

    endpoints = scraper.get_endpoints()
    file_path = scraper.setup_filepath()
    events = ['US CPI m/m', 'US Factory Orders m/m', 'US Advance GDP q/q', 'US Core CPI m/m', 'US Industrial Production m/m',
            'US Capacity Utilization Rate', 'US Trade Balance', 'US Unemployment Rate', 'CA Unemployment Rate',
            'CA Employment Change', 'EZ Consumer Confidence', 'US ISM Services PMI', 'US Factory Orders m/m',
            'US Retail Sales m/m', 'US Non-Farm Employment Change', 'US Average Hourly Earnings m/m', 
            'US Natural Gas Storage', 'US Gasoline Inventories', 'US Distillate Inventories', 
            'US Crude Oil Inventories', 'US ISM Manufacturing PMI']
    

    scraper.scrape_calenda_events(endpoints, file_path, events)
