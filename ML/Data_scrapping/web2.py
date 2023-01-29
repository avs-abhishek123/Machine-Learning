from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
from selenium.webdriver.common.by import By
# What you enter here will be searched for in
# Google Images
queries = [
    "elon_m",
    "elon_tesla",
    "elon_m_tesla",
    "elon_musk_tesla",
    "elon_msk_tesla",
    "elon_tesla_musk",
    "elon_teslamusk",
    "elon_musktesla",
    "elon_musk_",
    "elon_musk_tesla_",
    "elon_musk_t",
    "elon_musk_tsla",
    "elon_musk__",
    "elon_msk",
    "elon_mk",
    "elon_m",
]
for query in queries:
    # Creating a webdriver instance
    driver = webdriver.Chrome('C:/Users/MSI/Desktop/Machine-Learning/chromedriver.exe')

    # Maximize the screen
    driver.maximize_window()

    # Open Google Images in the browser
    driver.get('https://images.google.com/')

    # Finding the search box
    # box = driver.find_element(By.xpath,'//*[@id="sbtc"]/div/div[2]/input')
    box = driver.find_element("xpath", '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
    # Type the search query in the search box
    box.send_keys(query)

    # Pressing enter
    box.send_keys(Keys.ENTER)

    # Function for scrolling to the bottom of Google
    # Images results
    def scroll_to_bottom():

        last_height = driver.execute_script('\
        return document.body.scrollHeight')

        while True:
            driver.execute_script('\
            window.scrollTo(0,document.body.scrollHeight)')

            # waiting for the results to load
            # Increase the sleep time if your internet is slow
            time.sleep(10)

            new_height = driver.execute_script('\
            return document.body.scrollHeight')

            # click on "Show more results" (if exists)
            try:
                driver.find_element(By.css_selector,".YstHxe input").click()

                # waiting for the results to load
                # Increase the sleep time if your internet is slow
                time.sleep(0.2)

            except:
                pass

            # checking if we have reached the bottom of the page
            if new_height == last_height:
                break

            last_height = new_height


    # Calling the function

    # NOTE: If you only want to capture a few images,
    # there is no need to use the scroll_to_bottom() function.

    storage_location = "C:/Users/MSI/Desktop/Machine-Learning/images/final_images_elon/" + query +"/"

    if not os.path.exists(storage_location):
        os.makedirs(storage_location)

    # Loop to capture and save each image
    # j = 1
    # while j<=10:
    """
    scroll_to_bottom()
    for i in range(412, 700):

        # range(1, 50) will capture images 1 to 49 of the search results
        # You can change the range as per your need.
        try:

        # XPath of each image
            img = driver.find_element("xpath", 
                '//*[@id="islrg"]/div[1]/div[' +
            str(i) + ']/a[1]/div[1]/img')
            # //*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img
            # //*[@id="islrg"]/div[1]/div[2]/a[1]/div[1]/img
            # Enter the location of folder in which
            # the images will be saved
            img.screenshot(storage_location +
                        query + '_' + str(i) + '.png')
            # Each new screenshot will automatically
            # have its name updated

            # Just to avoid unwanted errors
            time.sleep(0.2)

        except:
            
            # if we can't find the XPath of an image,
            # we skip to the next image
            continue
        # j+=1
    # scroll_to_bottom()
    """

    for i in range(1, 390):

        # range(1, 50) will capture images 1 to 49 of the search results
        # You can change the range as per your need.
        try:

        # XPath of each image
            img = driver.find_element("xpath", 
                '//*[@id="islrg"]/div[1]/div[' +
            str(i) + ']/a[1]/div[1]/img')
            # //*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img
            # //*[@id="islrg"]/div[1]/div[2]/a[1]/div[1]/img
            # Enter the location of folder in which
            # the images will be saved
            img.screenshot(storage_location +
                        query + '_' + str(i) + '.png')
            # Each new screenshot will automatically
            # have its name updated

            # Just to avoid unwanted errors
            time.sleep(0.2)

        except:
            
            # if we can't find the XPath of an image,
            # we skip to the next image
            continue
        # j+=1
    # scroll_to_bottom()

    # Finally, we close the driver
    driver.close()
