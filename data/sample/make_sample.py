"""Generates a small synthetic twcs.csv-shaped sample so the pipeline can be
smoke-tested without downloading the real 3M-row Kaggle file. NOT used for
actual results -- only for verifying the code runs end-to-end."""
import pandas as pd

rows = [
    # thread 1: fare dispute
    dict(tweet_id=1, author_id="cust1", inbound=True, created_at="2023-01-01",
         text="@Uber_Support I was charged $47 for a ride that should have been $22, this is ridiculous",
         response_tweet_id="2", in_response_to_tweet_id=None),
    dict(tweet_id=2, author_id="Uber_Support", inbound=False, created_at="2023-01-01",
         text="@cust1 Sorry to hear that! Please DM us your trip ID and we'll take a look at the fare right away.",
         response_tweet_id=None, in_response_to_tweet_id=1),

    # thread 2: lost item
    dict(tweet_id=3, author_id="cust2", inbound=True, created_at="2023-01-02",
         text="@Uber_Support left my phone in the car after my ride this morning, how do I get it back?",
         response_tweet_id="4", in_response_to_tweet_id=None),
    dict(tweet_id=4, author_id="Uber_Support", inbound=False, created_at="2023-01-02",
         text="@cust2 No worries, please use the Lost Item option in your trip history in the app to contact your driver.",
         response_tweet_id=None, in_response_to_tweet_id=3),

    # thread 3: safety concern
    dict(tweet_id=5, author_id="cust3", inbound=True, created_at="2023-01-03",
         text="@Uber_Support my driver was driving so recklessly I felt unsafe the entire ride, swerving through traffic",
         response_tweet_id="6", in_response_to_tweet_id=None),
    dict(tweet_id=6, author_id="Uber_Support", inbound=False, created_at="2023-01-03",
         text="@cust3 We take this very seriously. Please DM us your trip details so we can investigate immediately.",
         response_tweet_id=None, in_response_to_tweet_id=5),

    # thread 4: driver behavior
    dict(tweet_id=7, author_id="cust4", inbound=True, created_at="2023-01-04",
         text="@Uber_Support driver was so rude to me, refused to help with my bags and was on the phone the whole time",
         response_tweet_id="8", in_response_to_tweet_id=None),
    dict(tweet_id=8, author_id="Uber_Support", inbound=False, created_at="2023-01-04",
         text="@cust4 That's not the experience we want for you. Please DM your trip ID so we can follow up with the driver.",
         response_tweet_id=None, in_response_to_tweet_id=7),

    # thread 5: account access
    dict(tweet_id=9, author_id="cust5", inbound=True, created_at="2023-01-05",
         text="@Uber_Support I can't log into my account, it keeps saying my password is wrong even after resetting",
         response_tweet_id="10", in_response_to_tweet_id=None),
    dict(tweet_id=10, author_id="Uber_Support", inbound=False, created_at="2023-01-05",
         text="@cust5 Sorry about that! Please DM the email on your account and we'll help you regain access.",
         response_tweet_id=None, in_response_to_tweet_id=9),

    # thread 6: cancellation fee
    dict(tweet_id=11, author_id="cust6", inbound=True, created_at="2023-01-06",
         text="@Uber_Support got charged a cancellation fee but the driver never moved from their spot",
         response_tweet_id="12", in_response_to_tweet_id=None),
    dict(tweet_id=12, author_id="Uber_Support", inbound=False, created_at="2023-01-06",
         text="@cust6 Sorry for the trouble! Please DM your trip ID and we'll review the cancellation fee.",
         response_tweet_id=None, in_response_to_tweet_id=11),

    # thread 7: general inquiry
    dict(tweet_id=13, author_id="cust7", inbound=True, created_at="2023-01-07",
         text="@Uber_Support do you guys offer student discounts?",
         response_tweet_id="14", in_response_to_tweet_id=None),
    dict(tweet_id=14, author_id="Uber_Support", inbound=False, created_at="2023-01-07",
         text="@cust7 We don't have a standard student discount, but keep an eye on promotions in the app!",
         response_tweet_id=None, in_response_to_tweet_id=13),

    # thread 8: app technical issue
    dict(tweet_id=15, author_id="cust8", inbound=True, created_at="2023-01-08",
         text="@Uber_Support the app keeps crashing every time I try to request a ride, using latest iOS",
         response_tweet_id="16", in_response_to_tweet_id=None),
    dict(tweet_id=16, author_id="Uber_Support", inbound=False, created_at="2023-01-08",
         text="@cust8 Sorry about that! Please try reinstalling the app, and DM us your device model if it persists.",
         response_tweet_id=None, in_response_to_tweet_id=15),

    # thread 9: another fare dispute (for retrieval testing)
    dict(tweet_id=17, author_id="cust9", inbound=True, created_at="2023-01-09",
         text="@Uber_Support why was I charged double the quoted price for my ride to the airport",
         response_tweet_id="18", in_response_to_tweet_id=None),
    dict(tweet_id=18, author_id="Uber_Support", inbound=False, created_at="2023-01-09",
         text="@cust9 We understand the concern. Please DM your trip ID so we can review the fare breakdown.",
         response_tweet_id=None, in_response_to_tweet_id=17),

    # thread 10: refund request
    dict(tweet_id=19, author_id="cust10", inbound=True, created_at="2023-01-10",
         text="@Uber_Support I want a refund, my driver cancelled on me twice in a row and I was late to work",
         response_tweet_id="20", in_response_to_tweet_id=None),
    dict(tweet_id=20, author_id="Uber_Support", inbound=False, created_at="2023-01-10",
         text="@cust10 So sorry about that! Please DM your trip details and we'll look into a refund for you.",
         response_tweet_id=None, in_response_to_tweet_id=19),
]

df = pd.DataFrame(rows)
df.to_csv("data/sample/twcs_sample.csv", index=False)
print(f"Wrote {len(df)} synthetic rows to data/sample/twcs_sample.csv")
