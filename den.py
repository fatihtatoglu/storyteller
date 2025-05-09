import requests
import os
import shutil

urls = {
    # "shakespeare_hamlet": "https://www.gutenberg.org/cache/epub/27761/pg27761.txt",
    # "shakespeare_macbeth": "https://www.gutenberg.org/cache/epub/1533/pg1533.txt",
    # "shakespeare_romeo_and_juliet": "https://www.gutenberg.org/cache/epub/1513/pg1513.txt",
    # "shakespeare_a_midsummer_nights_dream": "https://www.gutenberg.org/cache/epub/1514/pg1514.txt",
    # "shakespeare_othello": "https://www.gutenberg.org/cache/epub/1531/pg1531.txt",
    # "shakespeare_much_ado_about_nothing": "https://www.gutenberg.org/cache/epub/1519/pg1519.txt",
    # "shakespeare_king_lear": "https://www.gutenberg.org/cache/epub/1794/pg1794.txt",
    # "shakespeare_twelfth_night": "https://www.gutenberg.org/cache/epub/1526/pg1526.txt",
    # "shakespeare_the_tempest": "https://www.gutenberg.org/cache/epub/23042/pg23042.txt",
    # "shakespeare_the_merchant_of_venice": "https://www.gutenberg.org/cache/epub/1515/pg1515.txt",
    "dickens_a_tale_of_two_cities": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "dickens_a_christmas_carol": "https://www.gutenberg.org/cache/epub/24022/pg24022.txt",
    "dickens_great_expectations": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    "dickens_oliver_twist": "https://www.gutenberg.org/cache/epub/730/pg730.txt",
    "dickens_david_copperfield": "https://www.gutenberg.org/cache/epub/766/pg766.txt",
    "dickens_bleak_house": "https://www.gutenberg.org/cache/epub/1023/pg1023.txt",
    "dickens_hard_times": "https://www.gutenberg.org/cache/epub/786/pg786.txt",
    "dickens_little_dorrit": "https://www.gutenberg.org/cache/epub/963/pg963.txt",
    "dickens_our_mutual_friend": "https://www.gutenberg.org/cache/epub/883/pg883.txt",
    "dickens_the_pickwick_papers": "https://www.gutenberg.org/cache/epub/580/pg580.txt",
    "austen_pride_and_prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "austen_sense_and_sensibility": "https://www.gutenberg.org/cache/epub/161/pg161.txt",
    "austen_emma": "https://www.gutenberg.org/cache/epub/158/pg158.txt",
    "austen_persuasion": "https://www.gutenberg.org/cache/epub/105/pg105.txt",
    "austen_northanger_abbey": "https://www.gutenberg.org/cache/epub/121/pg121.txt",
    "austen_mansfield_park": "https://www.gutenberg.org/cache/epub/141/pg141.txt",
    "austen_lady_susan": "https://www.gutenberg.org/cache/epub/946/pg946.txt",
    "austen_love_and_freindship": "https://www.gutenberg.org/cache/epub/1212/pg1212.txt",
    "twain_adventures_of_huckleberry_finn": "https://www.gutenberg.org/cache/epub/76/pg76.txt",
    "twain_adventures_of_tom_sawyer": "https://www.gutenberg.org/cache/epub/74/pg74.txt",
    "twain_prince_and_pauper": "https://www.gutenberg.org/cache/epub/1837/pg1837.txt",
    "twain_connecticut_yankee_in_king_arthurs_court": "https://www.gutenberg.org/cache/epub/86/pg86.txt",
    "twain_puddnhead_wilson": "https://www.gutenberg.org/cache/epub/102/pg102.txt",
    "twain_diaries_of_adam_and_eve": "https://www.gutenberg.org/cache/epub/8525/pg8525.txt",
    "twain_life_on_the_mississippi": "https://www.gutenberg.org/cache/epub/245/pg245.txt",
    "twain_innocents_abroad_new_pilgrims_progress": "https://www.gutenberg.org/cache/epub/3176/pg3176.txt",
    "twain_mysterious_stranger": "https://www.gutenberg.org/cache/epub/3186/pg3186.txt",
    "shelley_the_last_man": "https://www.gutenberg.org/cache/epub/18247/pg18247.txt",
    "shelley_mathilda": "https://www.gutenberg.org/cache/epub/15238/pg15238.txt",
    "shelley_valperga_volume_1": "https://www.gutenberg.org/cache/epub/63337/pg63337.txt",
    "shelley_valperga_volume_2": "https://www.gutenberg.org/cache/epub/63338/pg63338.txt",
    "shelley_valperga_volume_3": "https://www.gutenberg.org/cache/epub/63339/pg63339.txt",
    "shelley_proserpine_and_midas": "https://www.gutenberg.org/cache/epub/6447/pg6447.txt",
    "shelley_the_dream": "https://www.gutenberg.org/cache/epub/56665/pg56665.txt",
    "melville_moby_dick": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "melville_bartleby_the_scrivener": "https://www.gutenberg.org/cache/epub/11231/pg11231.txt",
    "melville_benito_cereno": "https://www.gutenberg.org/cache/epub/15859/pg15859.txt",
    "melville_typee_peep_at_polynesian_life": "https://www.gutenberg.org/cache/epub/28656/pg28656.txt",
    "melville_the_confidence_man": "https://www.gutenberg.org/cache/epub/21816/pg21816.txt",
    "melville_pierre_or_the_ambiguities": "https://www.gutenberg.org/cache/epub/34970/pg34970.txt",
    "melville_redburn": "https://www.gutenberg.org/cache/epub/8118/pg8118.txt",
    "melville_white_jacket": "https://www.gutenberg.org/cache/epub/10712/pg10712.txt",
    "doyle_a_study_in_scarlet": "https://www.gutenberg.org/cache/epub/244/pg244.txt",
    "doyle_hound_of_the_baskervilles": "https://www.gutenberg.org/cache/epub/2852/pg2852.txt",
    "doyle_adventures_of_sherlock_holmes": "https://www.gutenberg.org/cache/epub/48320/pg48320.txt",
    "doyle_sign_of_four": "https://www.gutenberg.org/cache/epub/2097/pg2097.txt",
    "doyle_memoirs_of_sherlock_holmes": "https://www.gutenberg.org/cache/epub/834/pg834.txt",
    "doyle_return_of_sherlock_holmes": "https://www.gutenberg.org/cache/epub/108/pg108.txt",
    "doyle_the_lost_world": "https://www.gutenberg.org/cache/epub/139/pg139.txt",
    "doyle_valley_of_fear": "https://www.gutenberg.org/cache/epub/3289/pg3289.txt",
    "doyle_his_last_bow": "https://www.gutenberg.org/cache/epub/2350/pg2350.txt",
    "doyle_case_book_of_sherlock_holmes": "https://www.gutenberg.org/cache/epub/69700/pg69700.txt",
    "verne_around_the_world_in_eighty_days": "https://www.gutenberg.org/cache/epub/103/pg103.txt",
    "verne_twenty_thousand_leagues_under_the_sea": "https://www.gutenberg.org/cache/epub/164/pg164.txt",
    "verne_journey_to_the_center_of_the_earth": "https://www.gutenberg.org/cache/epub/18857/pg18857.txt",
    "verne_the_mysterious_island": "https://www.gutenberg.org/cache/epub/1268/pg1268.txt",
    "verne_from_the_earth_to_the_moon": "https://www.gutenberg.org/cache/epub/83/pg83.txt",
    "verne_five_weeks_in_a_balloon": "https://www.gutenberg.org/cache/epub/3526/pg3526.txt",
    "verne_in_search_of_castaways_children_of_captain_grant": "https://www.gutenberg.org/cache/epub/2083/pg2083.txt",
    "verne_miguel_strogoff": "https://www.gutenberg.org/cache/epub/1842/pg1842.txt",
    "verne_dick_sands_the_boy_captain": "https://www.gutenberg.org/cache/epub/9150/pg9150.txt",
}

if os.path.exists("books"):
    shutil.rmtree("books")

if os.path.exists("data_files"):
    shutil.rmtree("data_files")

os.mkdir("books")
os.mkdir("data_files")

for book, url in urls.items():
    response = requests.get(url)
    with open(f"./books/{book}.txt", "w", encoding="utf-8") as f:
        f.write(response.text)

for book, url in urls.items():
    with open(f"./books/{book}.txt", "r", encoding="utf-8") as r:
        text = r.read()

    gutenbergStartLine = "*** START OF THE PROJECT GUTENBERG EBOOK"
    gutenbergEndLine = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = text.find(gutenbergStartLine)
    end_idx = text.find(gutenbergEndLine)

    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]

    text = text.replace("\n\n", "\n").strip()

    with open(f"./data_files/{book}.txt", "w", encoding="utf-8") as w:
        w.write(text)
