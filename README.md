# Market-analysis
Market Analysis in Dublin
This data project has been used as a take-home assignment in the recruitment process for the data science positions at Airbnb.

## Assignment Overview

This project explores the dynamics of the Airbnb market in Dublin, focusing on guest searches, host responses, and market trends. The objective is to gain valuable insights into demand, supply patterns, and potential market gaps. The project involves thorough data exploration, visualization, and communication of key findings.
**There are 2 datasets**

searches.tsv - Contains a row for each set of searches that a user does for Dublin
contacts.tsv - Contains a row for every time that an assigned visitor makes an inquiry for a stay in a listing in Dublin
searches dataset contains the following columns:

**ds - Date of the search**
id_user - Alphanumeric user_id
ds_checkin - Date stamp of the check-in date of the search
ds_checkout - Date stamp of the check-out date of the search
n_searches - Number of searches in the search set
n_nights - The number of nights the search was for
n_guests_min - The minimum number of guests selected in a search set
n_guests_max - The maximum number of guests selected in a search set
origin_country - The country the search was from
filter_price_min - The value of the lower bound of the price filter, if the user used it
filter_price_max - The value of the upper bound of the price filter, if the user used it
filter_room_types - The room types that the user filtered by, if the user used the room_types filter
filter_neighborhoods - The neighborhoods types that the user filtered by, if the user used the neighborhoods filter
contacts dataset contains the following columns:

id_guest - Alphanumeric user_id of the guest making the inquiry
id_host - Alphanumeric user_id of the host of the listing to which the inquiry is made
id_listing - Alphanumeric identifier for the listing to which the inquiry is made
ts_contact_at - UTC timestamp of the moment the inquiry is made.
ts_reply_at - UTC timestamp of the moment the host replies to the inquiry, if so
ts_accepted_at - UTC timestamp of the moment the host accepts the inquiry, if so
ts_booking_at - UTC timestamp of the moment the booking is made, if so
ds_checkin - Date stamp of the check-in date of the inquiry
ds_checkout - Date stamp of the check-out date of the inquiry
n_guests - The number of guests the inquiry is for
n_messages - The total number of messages that were sent around this inquiry

## Key Findings

**Searches Dataset:**
- High null values in 'neighborhood.'
- Majority prefer 1 or 2 guests.
- Positive skewness in numeric columns.
- Searches mainly in October.
- 14.25% have prices exceeding 600.
- Room type data may need cleaning.
- Top origin countries identified.
**Contacts Dataset:**
- Positive skewness, especially in guest count.
- Majority have less than 8 guests.
- Conversion rate considered.
- Seasonal check-in patterns revealed.
**Merged Datasets Analysis:**
- Comparative analysis.
- Price impact on acceptance visualized.
- Price ranges classified.
- Varying acceptance rates by country.
