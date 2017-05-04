import sys, os
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Web_App.settings")

import django
django.setup()

from prediction.models import Posts

def save_post_in_df(data_row):
    post = Posts()
    post.content = data_row[0]
    post.tag = data_row[1]
    post.save()

if __name__ == "__main__":

    if len(sys.argv) == 2:
        print("Reading from file" + str(sys.argv[1]))
        post_df = pd.read_csv(sys.argv[1], error_bad_lines=False)
        print(post_df)

        post_df.apply(
            save_post_in_df,
            axis=1
        )

        print("There are {} posts".format(Posts.objects.count()))

    else:
        print("Please, provide file path")