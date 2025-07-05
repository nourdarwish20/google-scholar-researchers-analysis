"""Scrape Lebanese University researcher profiles from Google Scholar.

Walks the organisation page on Google Scholar, opens every researcher profile
and records name, total citations, h-index and number of listed papers.

Output: profiles.csv (overwrites the committed snapshot).

Requires Google Chrome. Selenium >= 4.6 downloads a matching chromedriver
automatically, so no driver binary needs to be shipped with the repo.
"""
import csv
import random
import sys
import time

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ORG_URL = "https://scholar.google.com/citations?view_op=view_org&hl=en&org=9671583371665794735"
OUTPUT_CSV = "profiles.csv"
MAX_PAGES = 30
MISSING = "Not found"


def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--start-maximized")
    return webdriver.Chrome(options=options)


def scrape_profile(driver, wait, url):
    """Open a profile in a new tab, extract its metrics, then close the tab."""
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])
    driver.get(url)

    profile_data = {
        'name': MISSING,
        'h_index_total': MISSING,
        'citations_total': MISSING,
        'total_papers': MISSING,
    }

    try:
        profile_data['name'] = wait.until(
            EC.presence_of_element_located((By.ID, "gsc_prf_in"))
        ).text.strip()

        # Metrics table: row 1 = citations, row 2 = h-index; column 1 = "All"
        try:
            rows = driver.find_element(By.ID, "gsc_rsb_st").find_elements(By.TAG_NAME, "tr")
            if len(rows) > 1:
                profile_data['citations_total'] = rows[1].find_elements(By.TAG_NAME, "td")[1].text.strip()
            if len(rows) > 2:
                profile_data['h_index_total'] = rows[2].find_elements(By.TAG_NAME, "td")[1].text.strip()
        except Exception as e:
            print(f"Error extracting citation/h-index: {e}")

        # Click "Show more" until every paper is loaded, then count them
        try:
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                try:
                    show_more = driver.find_element(By.ID, "gsc_bpf_more")
                except NoSuchElementException:
                    break
                if not show_more.is_enabled():
                    break
                show_more.click()
                time.sleep(1.5)

            publications = driver.find_elements(By.CLASS_NAME, "gsc_a_tr")
            profile_data['total_papers'] = str(len(publications))
        except Exception as e:
            print(f"Error extracting total papers: {e}")

    except Exception as e:
        print(f"Error scraping profile: {e}")
    finally:
        driver.close()
        driver.switch_to.window(driver.window_handles[0])

    return profile_data


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    driver = create_driver()
    wait = WebDriverWait(driver, 15)
    driver.get(ORG_URL)
    time.sleep(2)

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Name', 'Profile URL', 'Total Citations', 'H-index (Total)', 'Total Papers']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            for current_page in range(1, MAX_PAGES + 1):
                print(f"\n--- Processing page {current_page} ---")

                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.gs_ai")))
                profile_links = driver.find_elements(By.CSS_SELECTOR, "h3.gs_ai_name a")
                print(f"Found {len(profile_links)} profiles on this page")

                for idx, link in enumerate(profile_links, 1):
                    try:
                        profile_url = link.get_attribute("href")
                        if not profile_url:
                            continue

                        print(f"\nProcessing profile {idx}: {profile_url}")
                        data = scrape_profile(driver, wait, profile_url)
                        print(f"{data['name']} | h-index: {data['h_index_total']} | "
                              f"Citations: {data['citations_total']} | Papers: {data['total_papers']}")

                        writer.writerow({
                            'Name': data['name'],
                            'Profile URL': profile_url,
                            'Total Citations': data['citations_total'],
                            'H-index (Total)': data['h_index_total'],
                            'Total Papers': data['total_papers'],
                        })

                        # Be polite to Google Scholar
                        time.sleep(random.uniform(2, 4))

                    except Exception as e:
                        print(f"Failed to process profile: {e}")

                try:
                    next_button = wait.until(EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "button[aria-label='Next']")))
                    if 'disabled' in next_button.get_attribute("class"):
                        print("Reached last page")
                        break
                    next_button.click()
                    wait.until(EC.staleness_of(profile_links[0]))
                    time.sleep(random.uniform(3, 5))
                except Exception as e:
                    print(f"Pagination error: {e}")
                    break

        finally:
            driver.quit()
            print(f"\nScraping complete! Data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
