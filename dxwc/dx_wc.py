import argparse
import re
import itertools
import os
import glob
import random
import hashlib
import multiprocessing

import pandas as pd

from tqdm import tqdm
from tqdm.auto import tqdm 

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import imageio

tqdm.pandas()



ACTIVITY_FILE = "+dx-results/+tuning/git/Detailed_Model-git.csv"
MESSAGES_FILE = "+dx-results/+tuning/git/Commit_Messages-git.csv"

BLACKLIST = """
src main test java ci jobs resources dockerfile service exception model api 
paymenthub paymentshub payhub config configuration app application 
controller santander java main util utils enum json payment payments impl
data doc variable
""".split() +  stopwords.words('english')

def any_case_split(identifier):
    if not identifier:
        return []
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_|\s+|$)', identifier)
    return [m.group(0).replace("_", "").strip() for m in matches]


lemmatizer = WordNetLemmatizer() 

def sanitize(text: str):
    return re.sub("[\W|_]+", " ", text)


def cleanup_word(w):
    c = w.lower()
    c = lemmatizer.lemmatize(c)
    return c

def cleanup(words):
    return [cleanup_word(s) for s in words if not s.lower() in BLACKLIST and not s.isdigit() and len(s) > 2]


def get_color_for_string(some_string):
    some_hash = hashlib.shake_128(some_string.encode("utf-8")).hexdigest(3)
    rgb = tuple(int(some_hash[i:i+2], 16) for i in (0, 2, 4))

    return "rgba(%d,%d,%d,%d)" % (*rgb, 1)

def hash_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return get_color_for_string(word)

def make_word_cloud(word_frequencies, max_words=20):

    wc = WordCloud(background_color="white", max_words=max_words, prefer_horizontal=1.1,
                   width=1600, height=900)
    # generate word cloud
    wc.generate_from_frequencies(word_frequencies)
    wc = wc.recolor(color_func=hash_color_func, random_state=3)
    
    return wc
    
    
def draw_word_cloud(wc, title=None, save_path=None, show=False):
    plt.figure(figsize=(16, 9))
    # show
    plt.imshow(wc, interpolation="bilinear", aspect='auto')
    plt.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if title:
        plt.title(title, loc='left')
    
    # store to file
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    if show:
        plt.show()
    else:
        plt.clf() 
        plt.close() 
    

def plot_last_x_wordclouds(clouds):
    fig,axes = plt.subplots(nrows = 2, ncols = 3, figsize=(40,18))

    for ax in axes.flatten():
        ax.axis('off')

    ##edit this line to include your own image ids
    for i,wc in enumerate(clouds):
        c = i 
        ax = axes[int(c / 3), int(c % 3)]
        ax.imshow(wc[0], interpolation="bilinear", aspect='auto')
        ax.set_title(f"Activity of {wc[1].year}-{wc[1].month}", loc="left")

    plt.tight_layout()
    plt.show()

def interpolate(from_val, to_val, how_many):
    inc =  float(to_val - from_val)/(how_many + 1)
    return [from_val + i * inc for i in range(1, how_many + 1)]
                
# (('integration', 1.0), 233, (338, 15), None, 'rgba(167,199,14,1)')
def morph(from_wc, to_wc, how_many):
    
    
    from_dict = { o[0][0] : o for o in from_wc.layout_ }
    to_dict   = { o[0][0] : o for o in   to_wc.layout_ }
    
    all_words = []
    all_words.extend(from_dict.keys())
    all_words.extend(to_dict.keys())
    
    morph_freq = dict()
    morph_font = dict()
    morph_posx = dict()
    morph_posy = dict()
    
    for w in all_words:
        
        random.seed(w)
        
        if w not in from_dict:
            from_dict[w] = ((w, 0), 0, (random.randint(0, 1600), random.randint(0, 900)), None, get_color_for_string(w))
        if w not in to_dict:
            to_dict[w] = ((w, 0), 0, (random.randint(0, 1600), random.randint(0, 900)), None, get_color_for_string(w))
        
        morph_freq[w] = interpolate(from_dict[w][0][1],
                                    to_dict[w][0][1], 
                                    how_many)
        morph_font[w] = interpolate(from_dict[w][1],
                                    to_dict[w][1], 
                                    how_many)
        
        morph_posx[w] = interpolate(from_dict[w][2][0],
                                    to_dict[w][2][0], 
                                    how_many)
        morph_posy[w] = interpolate(from_dict[w][2][1],
                                    to_dict[w][2][1], 
                                    how_many)
        

    layouts = [{((w, morph_freq[w][i]), morph_font[w][i], (morph_posx[w][i], morph_posy[w][i]), None, from_dict[w][4]
                ) for w in all_words} for i in range(0, how_many)]
    
    return layouts


def get_wordclouds(scopes, frequency_fn, output_location, frames_between_keys=10, max_words=100):
    clouds = []
    frequency_df = frequency_fn(scopes[0])
    frequency_df.to_csv(os.path.join(output_location, f"word_frequency_{scopes[0]}.csv"))
    clouds.append((make_word_cloud(frequency_df.to_dict(), max_words=max_words), scopes[0]))

    for scope in tqdm(scopes[1:], desc="Generating frames per scope"):

        from_wc = clouds[-1][0]
        frequency_df = frequency_fn(scope)
        frequency_df.to_csv(os.path.join(output_location, f"word_frequency_{scope}.csv"))
        to_wc = make_word_cloud(frequency_df.to_dict(), max_words=max_words)

        for m in morph(from_wc, to_wc, frames_between_keys):
            f = make_word_cloud({"foo":1})
            f.layout_ = m
            clouds.append((f, scope))

        clouds.append((to_wc, scope))
        
    return clouds
    
    

def save_wordclouds_frames(location, wordclouds):
    indx = 0
    prev_scope=wordclouds[0][1]
    for wc, scope in tqdm(wordclouds, desc=f"Saving frames to {location}"):
        if prev_scope != scope:
            indx = 0
            prev_scope = scope

        draw_word_cloud(wc, show=False,
                        title=f"Activity for {scope}",
                        save_path=os.path.join(location, f"wc-{scope}-{indx:05d}"))
        indx += 1


def make_gif_from_frames(output, frames_location):
    frames = []
    duration = []
    prev_scope = None
    for f in tqdm(sorted(glob.glob(f'{frames_location}/*.png')), desc="Adding frames to output"):
        scope = f.split("/")[-1].split(".")[0].split("-")[-2] 
        frames.append(imageio.imread(f))
        if prev_scope and prev_scope != scope:
            duration.append(3.)
        else:
            duration.append(.1)
        prev_scope = scope
        
    duration.append(3.)
    frames.append(frames[-1])

    imageio.mimwrite(output, frames, duration=duration)


def prepare_data(dx_project):
    activity_df = pd.read_csv(os.path.join(dx_project, ACTIVITY_FILE), 
                              encoding='unicode_escape', 
                              on_bad_lines='warn')

    activity_df['date']  = pd.to_datetime(activity_df['date'])
    activity_df["churn"] = activity_df["addedLines"] + activity_df["deletedLines"]

    activity_df["rootpath"] = activity_df.file.apply(lambda x: x.split("/")[0])
    activity_df['words'] = activity_df.file.progress_apply(
        lambda x: "/".join(x.split('.')[:-1])).progress_apply(
        lambda x: x.split("/")[1:]).progress_apply(
        lambda x: list(itertools.chain(*[any_case_split(xx) for xx in x]))).progress_apply(
        cleanup)

    activity_words_df = activity_df.explode("words").reset_index()
    activity_words_df = activity_words_df[["words", "churn", "date"]]
    activity_words_df["source"] = "files"

    commit_messages_df = pd.read_csv(os.path.join(dx_project, MESSAGES_FILE), 
                                     index_col=0, 
                                     encoding='unicode_escape',
                                     on_bad_lines='warn'
                                     )[["date", "message"]]

    commit_messages_df['date'] = pd.to_datetime(commit_messages_df.date)

    commit_activity_df = activity_df.groupby(["commit"])[["churn"]].sum().reset_index()

    commit_messages_df = pd.merge(left=commit_messages_df, right=commit_activity_df, how="inner", 
                                left_index=True, right_on="commit")
    commit_messages_df["words"] = commit_messages_df.message.progress_apply(
        sanitize).progress_apply(
        any_case_split).progress_apply(
        cleanup)
    commit_messages_df
    commit_words_df = commit_messages_df[["words", "churn", "date"]].explode("words")
    commit_words_df["source"] = "messsages" 

    words_df = pd.concat([activity_words_df, commit_words_df])

    return words_df



def get_relevant_words_frequencies(scope, _df):
     return _df[_df.source.str.contains(scope)][["words", "churn"]].copy().groupby("words")["churn"].sum()


def main():
    args = parse_args()

    words_df = prepare_data(dx_project=args.dx)
    
    os.makedirs(args.output, exist_ok=True)

    clouds = get_wordclouds(scopes=["messsages|files", "messsages", "files"], 
                            frequency_fn=lambda scope: get_relevant_words_frequencies(scope, words_df), 
                            output_location=args.output,
                            frames_between_keys=0, max_words=args.max_words)
    save_wordclouds_frames(location=args.output, wordclouds=clouds)
   

def parse_args():
    parser = argparse.ArgumentParser(description='Extract the dependencies from the`requirements.yml` files.')
    parser.add_argument(
        '--dx', type=str, required=False, default=os.getcwd(),
        help='location of the DX project (dx has already been ran, and `+dx-results` folder is available), '
             'defaults to current working directory')
    parser.add_argument(
        '--max-words', type=int, required=False, default=20,
        help='Number of words to be a added on a word cloud, sorted descending by the amount of activity (churn).')
    parser.add_argument(
        '--output', type=str, required=False, default=os.path.join(os.getcwd(), "+wordclouds"),
        help='the folder where the output files will be written, defaults to `./+wordclouds`')
    args = parser.parse_args()

    if not (os.path.exists(os.path.join(args.dx, ACTIVITY_FILE)) and
            os.path.exists(os.path.join(args.dx, MESSAGES_FILE))):
        print(f"Path provided as DX project ({args.dx}) does not contain the expected files {ACTIVITY_FILE} and {MESSAGES_FILE}.")
        sys.exit(2)

    return args


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()